import asyncio
import base64
import io
import os
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    get_job_context,
    room_io,
)
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import google, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from PIL import Image

load_dotenv(".env.local")


async def analyze_image_with_blip(image_bytes: bytes) -> str:
    """Fallback using BLIP model"""
    client = AsyncInferenceClient(api_key=os.environ["HUGGINGFACE_API_KEY"])
    try:
        result = await client.image_to_text(
            image=image_bytes, model="Salesforce/blip-image-captioning-base"
        )
        return result.generated_text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


async def analyze_image_with_qwen(image_bytes: bytes, timeout: int = 10) -> str:
    """Analyze image with Qwen3-VL with timeout protection"""
    print("> Starting Image Analysis")
    client = AsyncInferenceClient(api_key=os.environ["HUGGINGFACE_API_KEY"])

    img_base64 = base64.b64encode(image_bytes).decode()

    # Concise prompt for faster response
    prompt = """Briefly describe this screenshot:
    - Main applications/windows visible
    - What the user is doing
    - Key text or UI elements
    Be concise and specific."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                },
            ],
        }
    ]

    try:
        result = await asyncio.wait_for(
            client.chat.completions.create(
                model="Qwen/Qwen3-VL-8B-Instruct", messages=messages, max_tokens=300
            ),
            timeout=timeout,
        )

        if hasattr(result, "choices") and len(result.choices) > 0:
            return str(result.choices[0].message.content)
        return str(result)

    except asyncio.TimeoutError:
        print("Vision API timeout, using fallback")
        return await analyze_image_with_blip(image_bytes)
    except Exception as e:
        print(f"Error with Qwen3-VL: {e}")
        return await analyze_image_with_blip(image_bytes)


def frame_to_jpeg_bytes(frame: rtc.VideoFrame, max_dimension: int = 1280) -> bytes:
    """Convert frame to JPEG with size optimization"""
    print("> Converting frame to JPEG")

    argb = frame.convert(rtc.VideoBufferType.ARGB)
    img = Image.frombuffer(
        "RGBA",
        (argb.width, argb.height),
        bytes(argb.data),
        "raw",
        "ARGB",
        0,
        1,
    )

    img = img.convert("RGB")

    # Resize if too large (vision models don't need full resolution)
    # if max(img.size) > max_dimension:
    #     ratio = max_dimension / max(img.size)
    #     new_size = tuple(int(dim * ratio) for dim in img.size)
    #     img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=85,
        subsampling=0,
        optimize=True,
    )

    return buf.getvalue()


class Assistant(Agent):
    def __init__(self, session: AgentSession) -> None:
        print("> Setting Up Assistant")
        super().__init__(
            instructions=(
                "You are a helpful desktop assistant. "
                "Ask the user to share their screen so you can help them. "
                "When they ask about their screen, describe what is visible naturally."
            )
        )
        self._session = session
        self._latest_screen_frame: Optional[rtc.VideoFrame] = None
        self._room: Optional[rtc.Room] = None
        self._last_frame_time = 0.0
        self._frame_interval = 0.5  # Update every 500ms max
        self._processing_screen = False  # Prevent concurrent vision calls

    async def on_enter(self) -> None:
        """Initialize the assistant."""
        self._room = get_job_context().room
        print(f"> Assistant entered room: {self._room.name}")

        @self._room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.Participant,
        ):
            if (
                participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
                and publication.source == rtc.TrackSource.SOURCE_SCREENSHARE
                and track.kind == rtc.TrackKind.KIND_VIDEO
            ):
                print(f"> Video track detected from {participant.identity}")
                self._start_screen_stream(track)

    def _start_screen_stream(self, track: rtc.Track) -> None:
        """Start streaming screen with frame rate limiting"""
        stream = rtc.VideoStream(track)
        print("> Starting video track streaming...")

        async def read_stream():
            print("> Started reading stream")
            try:
                async for event in stream:
                    # Rate limit frame updates
                    current_time = asyncio.get_event_loop().time()
                    if current_time - self._last_frame_time >= self._frame_interval:
                        self._latest_screen_frame = event.frame
                        self._last_frame_time = current_time
            except asyncio.CancelledError:
                print("> Stream reading cancelled")

        asyncio.create_task(read_stream())

    async def _describe_screen(self) -> str:
        """Describe the current screen with memory cleanup"""
        print("> Describing screen")

        if not self._latest_screen_frame:
            return "No screen share available."

        try:
            frame_copy = self._latest_screen_frame
            image_bytes = frame_to_jpeg_bytes(frame_copy)

            # Clear reference to allow GC
            frame_copy = None

            caption = await analyze_image_with_qwen(image_bytes)
            print(f"> Vision result: {caption[:100]}...")

            return caption

        except Exception as e:
            print(f"Vision error: {e}")
            return "Unable to analyze the screen at this moment."

    async def _process_screen_query(self, user_message: str) -> None:
        """Process screen analysis without blocking the audio pipeline"""
        if self._processing_screen:
            print("> Already processing a screen query, skipping")
            return

        self._processing_screen = True
        try:
            description = await self._describe_screen()

            await self._session.generate_reply(
                instructions=f"""The user said: "{user_message}"
                I analyzed their screen and see: {description}
                Provide a helpful, natural response based on this screen content."""
            )
        except Exception as e:
            print(f"Error processing screen query: {e}")
            await self._session.generate_reply(
                instructions="I had trouble analyzing the screen. Could you describe what you're seeing?"
            )
        finally:
            self._processing_screen = False

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """Handle user speech completion - optimized to avoid blocking"""
        print(f"> Received User Speech: {new_message.content}")

        msg = "".join(str(chunk) for chunk in new_message.content)
        user_text = msg.lower()

        keywords = (
            "screen",
            "see",
            "showing",
            "display",
            "what's on",
            "describe",
            "looking at",
            "what do you see",
        )

        if any(keyword in user_text for keyword in keywords):
            if self._latest_screen_frame:
                # Process asynchronously to avoid blocking audio
                asyncio.create_task(self._process_screen_query(msg))
            else:
                await self._session.generate_reply(
                    instructions=f'The user said: "{msg}". Let them know you cannot see their screen yet and ask them to share it.'
                )


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    llm = google.LLM(model="gemini-2.5-flash-lite")
    print(f"> Started LLM (voice): {llm.model}")

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm=llm,
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    print(f"> Setup Session: {session}")

    assistant = Assistant(session)
    print(f"> Started assistant: {assistant.id}")

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                )
            ),
        ),
    )
    print(f"> Started session: {session}")

    await session.generate_reply(
        instructions="You are a helpful desktop assistant. Ask the user to share their screen so you can help them."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)

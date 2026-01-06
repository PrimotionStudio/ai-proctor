import asyncio
import base64
import io
import os

import requests
from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient, InferenceClient
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    get_job_context,
    room_io,
)
from livekit.agents.llm import ChatContext, ChatMessage, ImageContent
from livekit.plugins import google, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from PIL import Image

load_dotenv(".env.local")

HF_API_URL = (
    "https://router.huggingface.co/models/Salesforce/blip-image-captioning-large"
)
HF_HEADERS = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}"}


async def analyze_image_with_blip(image_bytes: bytes) -> str:
    """Fallback using BLIP model which supports image-to-text directly"""
    client = AsyncInferenceClient(api_key=os.environ["HUGGINGFACE_API_KEY"])

    try:
        result = await client.image_to_text(
            image=image_bytes, model="Salesforce/blip-image-captioning-base"
        )
        return result.generated_text
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


async def analyze_image_with_qwen(image_bytes: bytes) -> str:
    print("> Starting Image Analysis")
    client = AsyncInferenceClient(api_key=os.environ["HUGGINGFACE_API_KEY"])

    # Convert bytes to base64 for the API
    img_base64 = base64.b64encode(image_bytes).decode()

    # Create a proper prompt for image analysis
    prompt = """Please analyze this screenshot and describe:
    1. What applications/windows are open
    2. What the user appears to be doing
    3. Any text or UI elements you can identify
    4. The overall context of the activity

    Be detailed and specific about what you see."""

    # Use the conversational interface for vision-language models
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
        result = await client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct", messages=messages, max_tokens=500
        )

        print(f"> Hugging Face raw response: {result}")

        # Extract the response content
        if hasattr(result, "choices") and len(result.choices) > 0:
            return str(result.choices[0].message.content)
        else:
            return str(result)

    except Exception as e:
        print(f"Error with Qwen3-VL, trying fallback: {e}")
        # Fallback: try with a simpler model
        return await analyze_image_with_blip(image_bytes)


def frame_to_jpeg_bytes(frame: rtc.VideoFrame) -> bytes:
    print("> Starting Frame to Byte Conversion")

    # Convert frame to ARGB buffer
    argb = frame.convert(rtc.VideoBufferType.ARGB)

    # IMPORTANT: Tell PIL this is ARGB, not RGBA
    img = Image.frombuffer(
        "RGBA",
        (argb.width, argb.height),
        bytes(argb.data),
        "raw",
        "ARGB",
        0,
        1,
    )

    # Drop alpha
    img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=95,  # 👈 better text
        subsampling=0,  # 👈 critical for screenshots
        optimize=True,
    )

    b = buf.getvalue()
    print(f"> Finished Frame to Byte Conversion: {b}")
    return b


class Assistant(Agent):
    def __init__(
        self,
        session: AgentSession,
    ) -> None:
        print(">Setting Up Assistant")
        super().__init__(
            instructions=(
                "You are a helpful assistant."
                "Ask the user to share their screen and confirm once they have. "
                "When they ask about their screen, describe what is visible."
            )
        )
        self._session = session
        self._latest_screen_frame = None
        self._room = None

    async def on_enter(self) -> None:
        """Initialize the assistant."""
        self._room = get_job_context().room
        print(f">Assistant has entered room: {self._room.name}")

        @self._room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.Participant,
        ):
            """Event listener when media track appears or is subscribed."""
            print(f">Track subscribed: {track.name}")
            if (
                participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
                and publication.source == rtc.TrackSource.SOURCE_SCREENSHARE
                and track.kind == rtc.TrackKind.KIND_VIDEO
            ):
                print(f">Video track detected from {participant.identity}")
                self._start_screen_stream(track)

    def _start_screen_stream(self, track: rtc.Track):
        """Starts streaming screen to get the latest frame."""
        stream = rtc.VideoStream(track)
        print(">Starting video track streaming...")

        async def read_stream():
            print(">Started reading stream")
            try:
                async for event in stream:
                    self._latest_screen_frame = event.frame
            except asyncio.CancelledError:
                print(">An exception happened while reading stream")
                pass

        asyncio.create_task(read_stream())

    async def _describe_screen(self) -> str:
        print(f">Starting Describe Last Screen Frame: {self._latest_screen_frame}")

        if not self._latest_screen_frame:
            return "No screen share available."

        try:
            print("> Sending frame to QWEN vision model")
            image_bytes = frame_to_jpeg_bytes(self._latest_screen_frame)
            caption = await analyze_image_with_qwen(image_bytes)

            print(f"> QWEN caption: {caption}")
            return caption

        except Exception as e:
            print(f"Vision error: {e}")
            return "Unable to analyze the screen."

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        print(f">Received User Speech: {new_message.content}")

        msg = ""
        for chunk in new_message.content:
            msg += f"{chunk}"

        user_text = msg.lower()
        keywords = (
            "screen",
            "see",
            "showing",
            "display",
            "what's on",
            "describe",
            "looking at",
        )
        if any(keyword in user_text for keyword in keywords):
            if self._latest_screen_frame:
                print(">Trying to get screen description")
                description = await self._describe_screen()
                print(f">Gotten Screen Description: {description}")
                await self._session.generate_reply(
                    instructions=f"""The user said: "{msg}"
                    I analyzed their screen and see: {description}
                    Provide a helpful, natural response based on this screen content."""
                )


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    llm = google.LLM(model="gemini-2.5-flash-lite")
    print(f">Started llm (voice): {llm.model}")

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm=llm,
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    print(f">Setup Session: {session}")

    assistant = Assistant(session)
    print(f">Started assistant: {assistant.id}")

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
    print(f">Started session: {session}")

    await session.generate_reply(
        instructions="You are a helpful desktop assistant. Ask the user to share their screen so you can help them."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)

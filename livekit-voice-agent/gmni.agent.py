import asyncio
import base64
import concurrent.futures  # Added for CPU offloading
import io
import os

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

# Global thread pool for image processing to prevent audio lag
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


async def analyze_image_with_qwen(image_bytes: bytes) -> str:
    """Optimized vision analysis"""
    client = AsyncInferenceClient(api_key=os.environ["HUGGINGFACE_API_KEY"])
    img_base64 = base64.b64encode(image_bytes).decode()

    prompt = "Describe the user's screen briefly. Focus on open apps and main activity."

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
        # Reduced max_tokens to 200 for faster response
        result = await client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct", messages=messages, max_tokens=200
        )
        return str(result.choices[0].message.content)
    except Exception as e:
        print(f"Vision error: {e}")
        return (
            "The screen shows various windows, but I'm having trouble seeing details."
        )


def _sync_frame_to_jpeg(frame: rtc.VideoFrame) -> bytes:
    """CPU-intensive task moved to a separate thread"""
    # Convert directly to RGB if possible, or RGBA then drop alpha
    rgba = frame.convert(rtc.VideoBufferType.RGBA)
    img = Image.frombuffer("RGBA", (rgba.width, rgba.height), bytes(rgba.data))
    img = img.convert("RGB")

    buf = io.BytesIO()
    # Lower quality slightly (80) to reduce network payload size/latency
    img.save(buf, format="JPEG", quality=80, optimize=True)
    return buf.getvalue()


class Assistant(Agent):
    def __init__(self, session: AgentSession) -> None:
        super().__init__(
            instructions="You are a helpful assistant. Use the screen descriptions provided to help the user."
        )
        self._session = session
        self._latest_screen_frame = None
        self._room = None

    async def on_enter(self) -> None:
        self._room = get_job_context().room

        @self._room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if (
                publication.source == rtc.TrackSource.SOURCE_SCREENSHARE
                and track.kind == rtc.TrackKind.KIND_VIDEO
            ):
                self._start_screen_stream(track)

    def _start_screen_stream(self, track: rtc.Track):
        stream = rtc.VideoStream(track)

        async def read_stream():
            async for event in stream:
                self._latest_screen_frame = event.frame

        asyncio.create_task(read_stream())

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        user_text = "".join([str(c) for c in new_message.content]).lower()

        keywords = ("screen", "see", "showing", "display", "describe", "looking at")

        if any(k in user_text for k in keywords) and self._latest_screen_frame:
            # 1. Immediate Feedback (Optional: reduces perceived lag)
            # await self._session.say("Let me take a look at your screen...")

            # 2. Offload image processing to a thread so audio doesn't stutter
            loop = asyncio.get_event_loop()
            image_bytes = await loop.run_in_executor(
                _executor, _sync_frame_to_jpeg, self._latest_screen_frame
            )

            # 3. Vision Analysis
            description = await analyze_image_with_qwen(image_bytes)

            # 4. Generate response with the context
            await self._session.generate_reply(
                instructions=f"User asked: {user_text}. Screen Analysis: {description}. Respond naturally."
            )


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    # Using Gemini 2.0 Flash (Fastest stable version for live interaction)
    llm = google.LLM(model="gemini-2.5-flash")

    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm=llm,
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    assistant = Assistant(session)
    await session.start(room=ctx.room, agent=assistant)
    await session.generate_reply(
        instructions="Introduce yourself and ask the user to share their screen."
    )


if __name__ == "__main__":
    agents.cli.run_app(server)

import asyncio
import base64
import os

from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient

load_dotenv()


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


async def analyze_image_with_blip(image_bytes: bytes) -> str:
    """Fallback using BLIP model which supports image-to-text directly"""
    client = AsyncInferenceClient(api_key=os.environ["HUGGINGFACE_API_KEY"])

    try:
        result = await client.image_to_text(
            image=image_bytes, model="Salesforce/blip-image-captioning-base"
        )
        return str(result.generated_text)
    except Exception as e:
        return f"Error analyzing image: {str(e)}"


async def main():
    # Read the image file
    image_path = "my_image.png"  # Change this to your image filename

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        print(f"Loaded image: {image_path} ({len(image_bytes)} bytes)")

        # Get the analysis
        analysis = await analyze_image_with_qwen(image_bytes)

        print(f"\n✅ Analysis: {analysis}")

    except FileNotFoundError:
        print(
            f"Error: Image file '{image_path}' not found. Please make sure the file exists."
        )
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())

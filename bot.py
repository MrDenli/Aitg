from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import Message, FSInputFile
from diffusers import StableDiffusionPipeline
import torch
import asyncio
import os

model_name = "runwayml/stable-diffusion-v1-5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Загрузка модели Stable Diffusion...")
pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipeline.to(device)

# api token

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
router = Router()

@router.message(F.text == "/start")
async def start_handler(message: Message):
    await message.answer("Привет! Напишите описание, и я сгенерирую изображение.")

@router.message()
async def generate_image_handler(message: Message):
    try:

        prompt = message.text
        image = pipeline(prompt).images[0]

        output_path = "output.png"
        image.save(output_path)

        photo = FSInputFile(output_path)
        await message.answer_photo(photo)

        if os.path.exists(output_path):
            os.remove(output_path)

    except Exception as e:
        await message.answer(f"Ошибка: {e}")

async def main():
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from huggingface_hub import login


#login

output_dir = "fine_tuned_model"
batch_size = 4
epochs = 5
learning_rate = 5e-6
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Загрузка предобученной модели CLIP...")
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
clip_model.to(device)

class CocoDataset(Dataset):
    def __init__(self, json_file):
        import json
        with open(json_file, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB").resize((224, 224))
        text = item["text"]
        return {"image": image, "text": text}


def collate_fn(batch):
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    return inputs


dataset = CocoDataset("data/coco_dataset.jsonl")

from torch.utils.data import Subset

subset_size = 4000
dataset = Subset(dataset, indices=list(range(subset_size)))

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(clip_model.parameters(), lr=learning_rate)

print("Начало обучения...")
clip_model.train()
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        images = batch["pixel_values"].to(device)
        texts = batch["input_ids"].to(device)

        outputs = clip_model(pixel_values=images, input_ids=texts)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(images.size(0)).to(device)

        loss = torch.nn.CrossEntropyLoss()(logits_per_image, labels) + \
               torch.nn.CrossEntropyLoss()(logits_per_text, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Эпоха {epoch + 1}, Шаг {step}, Потери: {loss.item()}")

os.makedirs(output_dir, exist_ok=True)
clip_model.save_pretrained(output_dir)
print(f"Модель успешно обучена и сохранена в {output_dir}!")

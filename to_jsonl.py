import os
import json

# Пути к данным
images_dir = "data/coco/images/train2017/"  
annotations_file = "data/coco/annotations/annotations/captions_train2017.json"
output_jsonl = "data/coco_dataset.jsonl" 

with open(annotations_file, "r") as f:
    annotations = json.load(f)

image_id_to_filename = {img["id"]: img["file_name"] for img in annotations["images"]}

os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
with open(output_jsonl, "w", encoding="utf-8") as jsonl_file:
    for ann in annotations["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        image_filename = image_id_to_filename.get(image_id)
        if not image_filename:
            continue

        image_path = os.path.join(images_dir, image_filename)

        if not os.path.exists(image_path):
            continue

        record = {"image": image_path, "text": caption}
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Данные сохранены в {output_jsonl}")

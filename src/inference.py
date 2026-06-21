# src/inference.py
import torch
from datasets import load_dataset
from PIL import ImageDraw
from dataset import TOKENIZER, normalize_bbox
from model import MimoDocumentModel

COLOR_MAP = {
    "HEADER": "green",
    "QUESTION": "blue",
    "ANSWER": "red",
    "OTHER": "gray",
    "O": "gray"
}

ID_TO_BIO = {
    0: "O",
    1: "B-HEADER", 2: "I-HEADER",
    3: "B-QUESTION", 4: "I-QUESTION",
    5: "B-ANSWER", 6: "I-ANSWER"
}

MAX_LEN = 512


def merge_bio_entities(tokens, predicted_labels, bboxes):
    merged_entities = []
    current_entity = None

    for token, label, bbox in zip(tokens, predicted_labels, bboxes):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        clean_token = token.replace("##", "")

        if label.startswith("B-"):
            if current_entity:
                merged_entities.append(current_entity)
            current_entity = {
                "label": label.replace("B-", ""),
                "text": clean_token,
                "bbox": list(bbox)
            }

        elif label.startswith("I-") or token.startswith("##"):
            if current_entity:
                current_entity["text"] += f" {clean_token}" if not token.startswith("##") else clean_token
                curr_box = current_entity["bbox"]
                current_entity["bbox"] = [
                    min(curr_box[0], bbox[0]),
                    min(curr_box[1], bbox[1]),
                    max(curr_box[2], bbox[2]),
                    max(curr_box[3], bbox[3])
                ]
            else:
                # Orphan I- tag: treat as start of entity
                current_entity = {
                    "label": label.replace("I-", ""),
                    "text": clean_token,
                    "bbox": list(bbox)
                }

        elif label == "O":
            if current_entity:
                merged_entities.append(current_entity)
                current_entity = None

    if current_entity:
        merged_entities.append(current_entity)

    return merged_entities


def run_inference_on_sample(model, test_sample, device):
    image = test_sample['image'].convert("RGB")
    width, height = image.size

    all_tokens = []
    all_orig_bboxes = []
    token_input_ids = []
    token_norm_bboxes = []

    for word, bbox in zip(test_sample['words'], test_sample['bboxes']):
        sub_tokens = TOKENIZER.tokenize(word)
        sub_token_ids = TOKENIZER.convert_tokens_to_ids(sub_tokens)
        normalized_box = normalize_bbox(bbox, width, height)

        for token, token_id in zip(sub_tokens, sub_token_ids):
            all_tokens.append(token)
            all_orig_bboxes.append(bbox)
            token_input_ids.append(token_id)
            token_norm_bboxes.append(normalized_box)

    all_tokens = all_tokens[:MAX_LEN]
    all_orig_bboxes = all_orig_bboxes[:MAX_LEN]
    token_input_ids = token_input_ids[:MAX_LEN]
    token_norm_bboxes = token_norm_bboxes[:MAX_LEN]

    tensor_ids = torch.tensor([token_input_ids], dtype=torch.long).to(device)
    tensor_bboxes = torch.tensor([token_norm_bboxes], dtype=torch.long).to(device)
    tensor_mask = torch.ones_like(tensor_ids).to(device)

    with torch.no_grad():
        logits = model(tensor_ids, tensor_bboxes, tensor_mask)

    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    predicted_bio_labels = [ID_TO_BIO.get(int(p), "O") for p in predictions]

    merged_entities = merge_bio_entities(all_tokens, predicted_bio_labels, all_orig_bboxes)

    draw = ImageDraw.Draw(image)
    for entity in merged_entities:
        color = COLOR_MAP.get(entity["label"], "gray")
        draw.rectangle(entity["bbox"], outline=color, width=2)

    return image, merged_entities


def run_inference():
    print("Loading test data and model weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data = load_dataset("nielsr/funsd")

    model = MimoDocumentModel(num_labels=7)
    model.load_state_dict(torch.load("mimo_funsd_weights.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    num_samples = 5
    for i in range(num_samples):
        test_sample = raw_data['test'][i]
        print(f"\n--- Sample {i+1}/{num_samples} ---")

        image, merged_entities = run_inference_on_sample(model, test_sample, device)

        print(f"Found {len(merged_entities)} entities:")
        for ent in merged_entities:
            print(f"  [{ent['label']}] \"{ent['text']}\"")

        output_filename = f"mimo_result_{i+1}.png"
        image.save(output_filename)
        print(f"✅ Saved to '{output_filename}'")


if __name__ == "__main__":
    run_inference()

# src/inference.py
import torch
from datasets import load_dataset
from PIL import ImageDraw
from dataset import TOKENIZER, normalize_bbox
from model import MimoDocumentModel

# Define our color palette
COLOR_MAP = {
    "HEADER": "green",
    "QUESTION": "blue",
    "ANSWER": "red",
    "OTHER": "gray",
    "O": "gray"
}

# NEW: We must include the B- and I- prefixes so the merger knows when to start/stop
ID_TO_BIO = {
    0: "O",
    1: "B-HEADER", 2: "I-HEADER", 
    3: "B-QUESTION", 4: "I-QUESTION",
    5: "B-ANSWER", 6: "I-ANSWER"
}

def merge_bio_entities(tokens, predicted_labels, bboxes):
    merged_entities = []
    current_entity = None

    for token, label, bbox in zip(tokens, predicted_labels, bboxes):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        clean_token = token.replace("##", "")

        # 1. Start a New Entity
        if label.startswith("B-"):
            if current_entity:
                merged_entities.append(current_entity)
            
            current_entity = {
                "label": label.replace("B-", ""), 
                "text": clean_token,
                "bbox": list(bbox) # Ensure it's a standard list
            }

        # 2. Expand the Current Entity
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
                current_entity = {
                    "label": label.replace("I-", ""),
                    "text": clean_token,
                    "bbox": list(bbox)
                }

        # 3. End of Entity
        elif label == "O":
            if current_entity:
                merged_entities.append(current_entity)
                current_entity = None 

    if current_entity:
        merged_entities.append(current_entity)

    return merged_entities


def run_inference():
    print("Loading test data and model weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    raw_data = load_dataset("nielsr/funsd")
    test_sample = raw_data['test'][0] 
    
    image = test_sample['image'].convert("RGB")
    width, height = image.size
    
    model = MimoDocumentModel(num_labels=7)
    model.load_state_dict(torch.load("mimo_funsd_weights.pt", map_location=device))
    model.to(device)
    model.eval() 

    input_ids = []
    input_bboxes = []
    word_to_token_map = [] 

    for word_idx, (word, bbox) in enumerate(zip(test_sample['words'], test_sample['bboxes'])):
        sub_tokens = TOKENIZER.tokenize(word)
        sub_token_ids = TOKENIZER.convert_tokens_to_ids(sub_tokens)
        normalized_box = normalize_bbox(bbox, width, height)
        
        for i, token_id in enumerate(sub_token_ids):
            input_ids.append(token_id)
            input_bboxes.append(normalized_box)
            if i == 0:
                word_to_token_map.append((word_idx, len(input_ids) - 1, bbox))

    tensor_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    tensor_bboxes = torch.tensor([input_bboxes], dtype=torch.long).to(device)
    tensor_mask = torch.ones_like(tensor_ids).to(device)

    print("Running document through the MIMO network...")
    with torch.no_grad(): 
        logits = model(tensor_ids, tensor_bboxes, tensor_mask)
    
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    # --- NEW: Prepare Data for the Merger ---
    raw_words = []
    raw_labels = []
    raw_bboxes = []

# --- The Clean Visualization Fix ---
    print("Drawing clean bounding boxes...")
    draw = ImageDraw.Draw(image)
    
    for word_idx, token_idx, original_bbox in word_to_token_map:
        predicted_id = predictions[token_idx]
        bio_label = ID_TO_BIO.get(predicted_id, "O")
        
        # Strip the B- and I- prefixes to get the core category
        category = bio_label.replace("B-", "").replace("I-", "")
        
        # MLOps Pro-Tip: Hide the background noise!
        # If the model predicts this word is just 'O' (Other), we don't draw a box at all.
        if category == "O":
            continue
            
        box_color = COLOR_MAP.get(category, "gray")
        
        # Draw the original, tight word box with the correct classification color
        draw.rectangle(original_bbox, outline=box_color, width=2)

    output_filename = "mimo_prediction_result_fixed.png"
    image.save(output_filename)
    print(f"✅ Inference complete! Open '{output_filename}' to see the actual masterpiece.")

if __name__ == "__main__":
    run_inference()
import torch
from datasets import load_dataset
from dataset import TOKENIZER, normalize_bbox
from model import MimoDocumentModel

# Our label map
ID_TO_BIO = {
    0: "O",
    1: "HEADER", 2: "HEADER", 
    3: "QUESTION", 4: "QUESTION",
    5: "ANSWER", 6: "ANSWER"
}

def extract_document_data():
    print("Loading model and FUNSD dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = MimoDocumentModel(num_labels=7)
    model.load_state_dict(torch.load("mimo_funsd_weights.pt", map_location=device))
    model.to(device)
    model.eval()
    
    raw_data = load_dataset("nielsr/funsd")
    
    # Loop through the first 3 unseen documents in the test set
    for doc_idx in range(3):
        print(f"\n{'='*40}")
        print(f"📄 ANALYZING DOCUMENT {doc_idx + 1}")
        print(f"{'='*40}")
        
        sample = raw_data['test'][doc_idx]
        width, height = sample['image'].size
        
        input_ids = []
        input_bboxes = []
        word_map = []

        # 1. Preprocess
        for word_idx, (word, bbox) in enumerate(zip(sample['words'], sample['bboxes'])):
            sub_tokens = TOKENIZER.tokenize(word)
            sub_token_ids = TOKENIZER.convert_tokens_to_ids(sub_tokens)
            norm_box = normalize_bbox(bbox, width, height)
            
            for i, token_id in enumerate(sub_token_ids):
                input_ids.append(token_id)
                input_bboxes.append(norm_box)
                if i == 0:
                    word_map.append((word_idx, len(input_ids) - 1))

        # 2. Tensor Prep
        tensor_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
        tensor_bboxes = torch.tensor([input_bboxes], dtype=torch.long).to(device)
        tensor_mask = torch.ones_like(tensor_ids).to(device)

        # 3. Model Inference
        with torch.no_grad():
            logits = model(tensor_ids, tensor_bboxes, tensor_mask)
        predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        # 4. Data Extraction & Grouping
        extracted_data = {"HEADER": [], "QUESTION": [], "ANSWER": []}
        
        for word_idx, token_idx in word_map:
            predicted_id = predictions[token_idx]
            category = ID_TO_BIO.get(predicted_id, "O")
            
            if category != "O":
                word_text = sample['words'][word_idx]
                extracted_data[category].append(word_text)
                
        # 5. Print the structured output!
        print("\n📌 HEADERS FOUND:")
        print(" ".join(extracted_data["HEADER"]) if extracted_data["HEADER"] else "None")
        
        print("\n❓ QUESTIONS FOUND:")
        print(" ".join(extracted_data["QUESTION"]) if extracted_data["QUESTION"] else "None")
        
        print("\n✅ ANSWERS FOUND:")
        print(" ".join(extracted_data["ANSWER"]) if extracted_data["ANSWER"] else "None")
        print("\n")

if __name__ == "__main__":
    extract_document_data()
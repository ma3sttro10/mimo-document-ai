from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")
def normalize_bbox(bbox,width,height):
    """
    Normalizes a bounding box to a 0-1000 scale.
    bbox: [x_min, y_min, x_max, y_max]
    """
    x_min , y_min , x_max , y_max = bbox
    x_min_norm = int(x_min/width * 1000)
    y_min_norm = int(y_min/height * 1000) 
    x_max_norm = int(x_max/width * 1000) 
    y_max_norm = int(y_max/height * 1000)
    # Ensure coordinates stay strictly within bounds (guardrails)
    return [
        max(0, min(x_min_norm, 1000)),
        max(0, min(y_min_norm, 1000)),
        max(0, min(x_max_norm, 1000)),
        max(0, min(y_max_norm, 1000))
    ]

class FunsdMimoDataset(Dataset):
    def __init__(self,hf_dataset,tokenizer,max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        raw_words = item['words']
        raw_bboxes = item['bboxes']
        raw_labels = item['ner_tags'] # <-- 1. Extract the labels
        
        width, height = item['image'].size

        input_ids = []
        input_bboxes = []
        labels = [] # <-- 2. Initialize label list

        for word, bbox, label in zip(raw_words, raw_bboxes, raw_labels):
            sub_tokens = self.tokenizer.tokenize(word)
            sub_token_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)
            normalized_box = normalize_bbox(bbox, width, height)
            
            for i, token_id in enumerate(sub_token_ids):
                input_ids.append(token_id)
                input_bboxes.append(normalized_box)
                
                # 3. Only the first sub-token gets the real label. 
                # The rest get -100 so the loss function ignores them.
                if i == 0:
                    labels.append(label)
                else:
                    labels.append(-100)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            input_bboxes = input_bboxes[:self.max_length]
            labels = labels[:self.max_length] # <-- 4. Truncate labels

        attention_mask = [1] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            input_bboxes += [[0, 0, 0, 0]] * padding_length
            attention_mask += [0] * padding_length
            labels += [-100] * padding_length # <-- 5. Pad empty space with -100

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "bbox": torch.tensor(input_bboxes, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long) # <-- 6. Return tensor
        }
# --- Quick Test ---
if __name__ == "__main__":
    # Quick sanity check test
    raw_data = load_dataset("nielsr/funsd")
    train_dataset = FunsdMimoDataset(raw_data['train'], TOKENIZER)
    
    # Test loading the very first sample
    sample_tensor_dict = train_dataset[0]
    print("Input IDs Shape:", sample_tensor_dict["input_ids"].shape)
    print("BBox Tensor Shape:", sample_tensor_dict["bbox"].shape)
    print("Attention Mask Shape:", sample_tensor_dict["attention_mask"].shape)
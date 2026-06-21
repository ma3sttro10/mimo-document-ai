# src/model.py
import torch
import torch.nn as nn
from transformers import LayoutLMForTokenClassification


class MimoDocumentModel(nn.Module):
    def __init__(self, num_labels=7):
        super().__init__()
        # LayoutLM is the production version of this project's architecture:
        # BERT + 2D spatial position embeddings, pre-trained on 11M documents.
        self.layoutlm = LayoutLMForTokenClassification.from_pretrained(
            "microsoft/layoutlm-base-uncased",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, input_ids, bbox, attention_mask):
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask
        )
        return outputs.logits


if __name__ == "__main__":
    dummy_input_ids = torch.randint(0, 30522, (2, 512))
    dummy_bbox = torch.randint(0, 1000, (2, 512, 4))
    dummy_mask = torch.ones((2, 512), dtype=torch.long)

    model = MimoDocumentModel(num_labels=7)
    out_logits = model(dummy_input_ids, dummy_bbox, dummy_mask)
    print(f"Output Matrix Shape: {out_logits.shape}")

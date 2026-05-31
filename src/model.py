# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class MimoDocumentModel(nn.Module):
    def __init__(self, num_labels=7):
        super(MimoDocumentModel, self).__init__()
        
        # 1. Load the core Transformer Engine (BERT)
        # We use bert-base-uncased which has a hidden dimension of 768
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        
        # 2. Define the Spatial Lookup Tables (Embeddings)
        # We need 1001 rows (since coordinates go from 0 to 1000)
        # and 768 columns (to match BERT's hidden dimension)
        self.x_embedding = nn.Embedding(num_embeddings=1001, embedding_dim=768)
        self.y_embedding = nn.Embedding(num_embeddings=1001, embedding_dim=768)
        
        # 3. The final classification head
        self.spatial_norm = nn.LayerNorm(768)
        # This will predict if a word is a Header, Question, Answer, or Other
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, bbox, attention_mask):
        # --- THE EMBEDDING PHASE ---
        
        # Extract the 4 coordinates from the bbox tensor: [Batch, Sequence, 4]
        x_min = bbox[:, :, 0]
        y_min = bbox[:, :, 1]
        x_max = bbox[:, :, 2]
        y_max = bbox[:, :, 3]
        
        # Look up the spatial vectors for each coordinate
        e_x_min = self.x_embedding(x_min)
        e_y_min = self.y_embedding(y_min)
        e_x_max = self.x_embedding(x_max)
        e_y_max = self.y_embedding(y_max)
        
        # Formula 1: Sum the coordinates to create a single layout vector
        raw_spatial = e_x_min + e_y_min + e_x_max + e_y_max
        # NEW: Normalize the spatial vector BEFORE fusion
        e_spatial = self.spatial_norm(raw_spatial)
        
        # 3. Finally, we fuse the clean spatial vector with the text
        e_text = self.bert.embeddings.word_embeddings(input_ids)
        e_fused = e_text + e_spatial
        
        # --- THE TRANSFORMER PHASE ---
        
        # Now, we bypass BERT's normal input (input_ids) and forcefully inject 
        # our custom fused vectors using `inputs_embeds`
        outputs = self.bert(
            inputs_embeds=e_fused,
            attention_mask=attention_mask
        )
        
        # Extract the final hidden states for every token in the sequence
        sequence_output = outputs.last_hidden_state
        
        # Pass them through our linear classifier to get final predictions
        logits = self.classifier(sequence_output)
        
        return logits

# --- Quick Architecture Test ---
if __name__ == "__main__":
    # Simulate a dummy batch (Batch Size = 2, Sequence Length = 512)
    dummy_input_ids = torch.randint(0, 30522, (2, 512))
    dummy_bbox = torch.randint(0, 1000, (2, 512, 4))
    dummy_mask = torch.ones((2, 512))
    
    # Initialize our custom PyTorch Model
    model = MimoDocumentModel(num_labels=4)
    
    # Push the dummy tensors through the network
    out_logits = model(dummy_input_ids, dummy_bbox, dummy_mask)
    
    print(f"Output Matrix Shape: {out_logits.shape}")
# src/train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
# Import our custom modules
from dataset import FunsdMimoDataset, TOKENIZER
from model import MimoDocumentModel

def train():
    # 1. Force PyTorch to use RTX GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Launching training on: {device}")

    # 2. Data Loading
    print("Loading FUNSD dataset...")
    raw_data = load_dataset("nielsr/funsd")
    
    # Initialize our dataset (assuming we added label extraction to it)
    train_dataset = FunsdMimoDataset(raw_data['train'], TOKENIZER)
    
    # The DataLoader automatically batches our 512x512 matrices and shuffles them
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 3. Model & Optimization Setup
    model = MimoDocumentModel(num_labels=7).to(device)
    
    # AdamW is the enterprise standard for Transformer models
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': 5e-5},
        {'params': model.x_embedding.parameters(), 'lr': 5e-4},
        {'params': model.y_embedding.parameters(), 'lr': 5e-4},
        {'params': model.spatial_norm.parameters(), 'lr': 5e-4},
        {'params': model.classifier.parameters(), 'lr': 5e-4}
    ])
    
    # The magic function: Notice how ignore_index=-100 is built-in by default, 
    # but we state it explicitly here for MLOps clarity.
    # Classes 1 through 6 (Headers, Questions, Answers) get a heavy weight of 1.5
    class_weights = torch.tensor([0.2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]).to(device)
    
    # 2. Inject the weights into the loss function
    loss_fn = CrossEntropyLoss(weight=class_weights, ignore_index=-100)
    # 4. The Training Loop (Epochs)
    epochs = 30
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    model.train() # Lock the model into training mode

    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move all tensors from RAM to the GPU VRAM
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # (Assuming you added 'labels' to the dataset output dictionary)
            labels = batch['labels'].to(device) 
            
            # Step A: Zero out the old gradients from the last batch
           

            # Step B: Forward Pass (Push data through the MIMO architecture)
            logits = model(input_ids, bbox, attention_mask)

            # Step C: Calculate Loss
            # PyTorch expects a 2D matrix for logits and 1D for labels, so we flatten them
            active_loss = attention_mask.view(-1) == 1
            # 2. APPLY THE MASK: Strip out all padding from the arrays
            active_logits = logits.view(-1, 7)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            # 3. Calculate pure signal loss
            loss = loss_fn(active_logits, active_labels)

            # Step D: Backward Pass (Calculate the gradients)
            loss.backward()

            # Step E: Update the model's weights
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f} ---")

    # 5. Save the weights for the FastAPI deployment
    torch.save(model.state_dict(), "mimo_funsd_weights.pt")
    print("✅ Model weights saved successfully. Ready for deployment.")

if __name__ == "__main__":
    train()
# src/train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup
from dataset import FunsdMimoDataset, TOKENIZER
from model import MimoDocumentModel


LABEL_NAMES = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]


def evaluate(model, loader, loss_fn, device, verbose=False):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, bbox, attention_mask)

            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, 7)[active_loss]
            active_labels = labels.view(-1)[active_loss]

            real_mask = active_labels != -100
            active_logits = active_logits[real_mask]
            active_labels = active_labels[real_mask]

            loss = loss_fn(active_logits, active_labels)
            total_loss += loss.item()

            preds = torch.argmax(active_logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(active_labels.cpu().tolist())

    model.train()
    avg_loss = total_loss / len(loader)
    total = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / total if total > 0 else 0.0

    if verbose:
        # Per-class: how many times was each class predicted vs how many times it truly appeared
        pred_counts = [0] * 7
        true_counts = [0] * 7
        class_correct = [0] * 7
        for p, l in zip(all_preds, all_labels):
            pred_counts[p] += 1
            true_counts[l] += 1
            if p == l:
                class_correct[l] += 1
        print("  Class breakdown (predicted / actual / recall):")
        for i, name in enumerate(LABEL_NAMES):
            recall = class_correct[i] / true_counts[i] if true_counts[i] > 0 else 0.0
            print(f"    {name:<12} pred={pred_counts[i]:5d}  actual={true_counts[i]:5d}  recall={recall:.2f}")

    return avg_loss, accuracy


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Launching training on: {device}")

    print("Loading FUNSD dataset...")
    raw_data = load_dataset("nielsr/funsd")

    train_dataset = FunsdMimoDataset(raw_data['train'], TOKENIZER)
    val_dataset = FunsdMimoDataset(raw_data['test'], TOKENIZER)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = MimoDocumentModel(num_labels=7).to(device)

    # Single learning rate for all parameters — LayoutLM is already pre-trained,
    # so all layers just need gentle fine-tuning at the same rate.
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # Inverse-frequency weights derived from actual FUNSD val distribution:
    # O=2358, B-HDR=119, I-HDR=255, B-Q=1065, I-Q=1478, B-A=809, I-A=2485
    class_weights = torch.tensor([0.45, 5.0, 4.2, 1.0, 0.72, 1.32, 0.43]).to(device)
    loss_fn = CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    epochs = 30
    early_stop_patience = 5
    epochs_without_improvement = 0

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_val_acc = 0.0
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            bbox = batch['bbox'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Step A: Zero gradients before the forward pass
            optimizer.zero_grad()

            # Step B: Forward pass
            logits = model(input_ids, bbox, attention_mask)

            # Step C: Compute loss on non-padding tokens only
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, 7)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fn(active_logits, active_labels)

            # Step D: Backward pass
            loss.backward()

            # Clip gradients to prevent exploding gradient issues on small dataset
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Step E: Update weights
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        verbose = (epoch + 1) % 5 == 0  # print class breakdown every 5 epochs
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, verbose=verbose)

        print(
            f"--- Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.3f} ---"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "mimo_funsd_weights.pt")
            print(f"  ✅ New best model saved (val_acc={val_acc:.3f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement}/{early_stop_patience} epochs")
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print("Training complete. Best weights saved to mimo_funsd_weights.pt")


if __name__ == "__main__":
    train()

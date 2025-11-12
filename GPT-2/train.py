import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import GPTConfig, GPT

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print(device)

# Global tokenizer (must match vocab_size in GPTConfig)
ENCODING_NAME = "gpt2"
enc = tiktoken.get_encoding(ENCODING_NAME)

def load_and_split_data(file_path, train_ratio=0.9):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = torch.tensor(enc.encode(text), dtype=torch.long)

    n = int(train_ratio * len(tokens))
    train_tokens = tokens[:n]
    val_tokens = tokens[n:]

    print(f"Loaded {len(tokens):,} tokens | train: {len(train_tokens):,}, val: {len(val_tokens):,}")
    return train_tokens, val_tokens

@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    total_loss = 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
    model.train()
    return total_loss / (i + 1)


class GPTDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
        self.num_samples = (len(tokens) - 1) // block_size  # Non-overlapping

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return x, y


def create_dataloaders(train_tokens, val_tokens, block_size, batch_size):
    train_dataset = GPTDataset(train_tokens, block_size)
    val_dataset = GPTDataset(val_tokens, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader

def train_model(model, train_loader, val_loader, optimizer, device, epochs=5,
                eval_interval=200, save_dir="checkpoints",
                scheduler=None, max_eval_batches=50):

    os.makedirs(save_dir, exist_ok=True)
    global_step = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        progress = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (x, y) in enumerate(progress):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            global_step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

            if global_step % eval_interval == 0:
                train_loss = evaluate(model, train_loader, device, max_batches=max_eval_batches)
                val_loss   = evaluate(model, val_loader, device, max_batches=max_eval_batches)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                print(f"\nStep {global_step} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                ckpt_path = os.path.join(save_dir, f"step_{global_step}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "step": global_step
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete.")
    return train_losses, val_losses


model = GPT(GPTConfig())
model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# ==== 3. Create optimizer ====
epochs = 100  #

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95),weight_decay=0.1)

# === 2. Load and split data ===
train_tokens, val_tokens = load_and_split_data("../datasets/shakespeare/input.txt")

# === 4. Create dataloaders ===
block_size = 256
batch_size = 32
train_loader, val_loader = create_dataloaders(train_tokens, val_tokens, block_size, batch_size)
print(f"Batches per epoch: {len(train_loader)}")

# === 5. Create scheduler ===
max_iters = epochs * len(train_loader)
warmup_iters = max_iters // 10  # 10% warmup
scheduler = CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)


print(f"Total parameters: {count_parameters(model):,}")

# ==== 6. Train the model ====
train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=epochs,  # ← Use the variable
    eval_interval=200,
    scheduler=scheduler,
    save_dir="checkpoints"
)
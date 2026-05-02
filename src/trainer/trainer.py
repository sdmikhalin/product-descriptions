import torch
from torch.optim import AdamW
from tqdm import tqdm

class BlipTrainer:
    def __init__(self, model, train_loader, val_loader,
                 lr=5e-5, epochs=3, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.epochs = epochs
        self.device = device

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def fit(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            print(f"Epoch {epoch + 1}: train loss = {train_loss:.4f}")
        return self.model

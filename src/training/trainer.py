# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # Для логирования (опционально)

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: torch.device = torch.device('cpu'),
    checkpoint_path: str = 'models/music_lstm.pth',
):
    from pathlib import Path
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    """
    Тренировочный цикл с валидацией.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD, если есть
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)  # Уменьшает lr при плато
    
    writer = SummaryWriter()  # TensorBoard для графиков (запускай tensorboard --logdir=runs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits, _ = model(x)  # [batch, seq, vocab]
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        scheduler.step(val_loss)  # Уменьшить lr если val_loss не падает
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model at {checkpoint_path}")
    
    writer.close()
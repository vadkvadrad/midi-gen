# train.py

import torch
from pathlib import Path
from src.data.prepare import prepare_music_data
from src.models.lstm_improved import MusicLSTM
from src.training.trainer import train_model
from src.generation.sampler import generate_music, save_generated_midi

# Создаем необходимые директории
Path("models").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Данные
train_loader, val_loader, tokenizer = prepare_music_data(
    midi_dir='data/midi/',
    context_len=256,  # Уменьшено для работы с более короткими последовательностями
    stride=64,
    tokenizer_path='models/tokenizer.json'
)

# Модель
model = MusicLSTM(len(tokenizer), embed_size=256, hidden_size=512, num_layers=3).to(device)

# Обучение
train_model(model, train_loader, val_loader, epochs=50, device=device)

# Генерация (после обучения)
print("\nGenerating sample music...")
seed = next(iter(val_loader))[0][0].tolist()[:50]  # Первый сэмпл из val как seed
generated = generate_music(
    model, tokenizer, seed, 
    gen_length=300, 
    temperature=0.8, 
    device=device,
    context_len=256
)
save_generated_midi(generated, tokenizer, 'output/generated_music.mid')
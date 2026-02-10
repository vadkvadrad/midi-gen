#!/usr/bin/env python3
"""
Скрипт для генерации музыки из обученной модели.
"""

import torch
import argparse
from pathlib import Path
from src.models.lstm_improved import MusicLSTM
from src.generation.sampler import generate_music, save_generated_midi
from miditok import REMI

def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    device: torch.device
):
    """Загружает обученную модель и токенизатор."""
    # Загружаем токенизатор
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = REMI.from_file(tokenizer_path)
    vocab_size = len(tokenizer)
    
    # Создаем модель с правильным размером словаря
    model = MusicLSTM(
        vocab_size=vocab_size,
        embed_size=256,
        hidden_size=512,
        num_layers=3
    ).to(device)
    
    # Загружаем веса модели
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, tokenizer

def generate(
    model_path: str = "models/music_lstm.pth",
    tokenizer_path: str = "models/tokenizer.json",
    output_path: str = "output/generated_music.mid",
    seed_length: int = 50,
    gen_length: int = 500,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed_tokens: list = None,
):
    """
    Генерирует MIDI файл из обученной модели.
    
    Args:
        model_path: Путь к обученной модели
        tokenizer_path: Путь к токенизатору
        output_path: Путь для сохранения сгенерированного MIDI
        seed_length: Длина начальной последовательности (если seed_tokens не задан)
        gen_length: Длина генерируемой последовательности
        temperature: Температура для sampling (больше = более случайно)
        top_p: Nucleus sampling параметр
        seed_tokens: Начальная последовательность токенов (опционально)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Загружаем модель и токенизатор
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path, device)
    
    # Создаем seed последовательность
    if seed_tokens is None:
        # Простой seed: начинаем с Bar и Position токенов
        # Это базовые токены REMI, которые должны быть в словаре
        try:
            # Пытаемся найти базовые токены
            vocab = tokenizer.vocab
            bar_token = vocab.get('Bar_None', 0)
            pos_token = vocab.get('Position_0', 1)
            seed_tokens = [bar_token, pos_token] * (seed_length // 2)
        except:
            # Если не получилось, используем случайные токены из словаря
            import random
            vocab_size = len(tokenizer)
            seed_tokens = [random.randint(0, vocab_size - 1) for _ in range(seed_length)]
    
    print(f"Generating {gen_length} tokens with seed length {len(seed_tokens)}")
    
    # Генерируем музыку
    generated_tokens = generate_music(
        model=model,
        tokenizer=tokenizer,
        seed_sequence=seed_tokens,
        gen_length=gen_length,
        temperature=temperature,
        top_p=top_p,
        device=device,
        context_len=512
    )
    
    # Сохраняем результат
    save_generated_midi(generated_tokens, tokenizer, output_path)
    print(f"Generation complete! Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIDI music from trained model")
    parser.add_argument("--model", type=str, default="models/music_lstm.pth",
                        help="Path to trained model")
    parser.add_argument("--tokenizer", type=str, default="models/tokenizer.json",
                        help="Path to tokenizer")
    parser.add_argument("--output", type=str, default="output/generated_music.mid",
                        help="Output MIDI file path")
    parser.add_argument("--seed-length", type=int, default=50,
                        help="Length of seed sequence")
    parser.add_argument("--gen-length", type=int, default=500,
                        help="Length of generated sequence")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling parameter")
    
    args = parser.parse_args()
    
    generate(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        seed_length=args.seed_length,
        gen_length=args.gen_length,
        temperature=args.temperature,
        top_p=args.top_p
    )


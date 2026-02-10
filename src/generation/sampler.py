# src/generation/sampler.py

import torch
import torch.nn as nn
from miditok import REMI  # Твой tokenizer

def generate_music(
    model: nn.Module,
    tokenizer: REMI,
    seed_sequence: list[int],  # Начальные токены (из тестовых данных)
    gen_length: int = 500,     # Длина генерации
    temperature: float = 1.0,  # >1 = больше randomness, <1 = детерминировано
    top_p: float = 0.95,       # Nucleus sampling (лучше multinomial)
    device: torch.device = torch.device('cpu'),
    context_len: int = 512,    # Длина контекста для генерации
) -> list[int]:
    """
    Авторегрессивная генерация: предсказываем по одному токену.
    """
    model.eval()
    with torch.no_grad():
        generated = seed_sequence.copy()
        hidden = None
        
        for _ in range(gen_length):
            # Берем последние context_len токенов (или меньше, если еще не накопили)
            context = generated[-context_len:] if len(generated) >= context_len else generated
            input_tensor = torch.tensor([context]).to(device)
            logits, hidden = model(input_tensor, hidden=hidden)
            
            # Последний logit
            logits = logits[0, -1] / temperature
            
            # Top-p sampling (актуально в 2026 для разнообразия)
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=0)
            mask = cumulative_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()  # Нормализация
            
            next_token = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
            generated.append(next_token)
    
    return generated

def save_generated_midi(tokens: list[int], tokenizer: REMI, output_path: str = 'generated.mid'):
    """
    Декодируем токены в MIDI.
    """
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # miditok decode возвращает Score (symusic)
    midi = tokenizer.decode(tokens)  # Из miditok
    midi.dump(output_path)  # Score.dump() сохраняет MIDI файл
    print(f"Saved MIDI to {output_path}")

# Пример
# generated_tokens = generate_music(model, tokenizer, seed=[tokenizer['Bar'], tokenizer['Position_0'], ...])
# save_generated_midi(generated_tokens, tokenizer)
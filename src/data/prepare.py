# src/data/prepare.py

import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from miditok import REMI, TokenizerConfig
from symusic import Score
from tqdm import tqdm

# ────────────────────────────────────────────────
#   Конфигурация токенизатора (REMI — золотой стандарт 2025+)
# ────────────────────────────────────────────────

def create_remi_tokenizer(
    num_velocities: int = 32,           # 32 уровня velocity → хороший баланс
    use_chords: bool = True,
    use_programs: bool = False,         # False = пианино-only / mono-instrument
    use_tempos: bool = False,
    use_time_signatures: bool = True,
    beat_res: Optional[Dict[int, Tuple[int, int]]] = None,  # Словарь {denominator: (start, end)}
    additional_bar_marker: bool = False,
) -> REMI:
    # Создаем базовую конфигурацию
    # НЕ передаём beat_res - используем значения по умолчанию из TokenizerConfig
    # Это избегает проблем с форматом beat_res в разных версиях miditok
    config = TokenizerConfig(
        num_velocities=num_velocities,
        use_chords=use_chords,
        use_programs=use_programs,
        use_tempos=use_tempos,
        use_time_signatures=use_time_signatures,
        additional_bar_marker=additional_bar_marker,
    )
    return REMI(config)


# ────────────────────────────────────────────────
#   Загрузка и токенизация MIDI-файлов
# ────────────────────────────────────────────────

def load_and_tokenize_midis(
    midi_dir: str | Path,
    tokenizer: REMI,
    max_files: Optional[int] = None,
    min_seq_len: int = 64,              # отбрасываем слишком короткие треки
) -> List[List[int]]:
    midi_paths = sorted(Path(midi_dir).glob("**/*.mid")) + \
                 sorted(Path(midi_dir).glob("**/*.midi"))
    
    if max_files is not None:
        midi_paths = midi_paths[:max_files]
    
    all_tokens = []
    skipped_short = 0
    token_lengths = []
    
    for path in tqdm(midi_paths, desc="Tokenizing MIDI files"):
        try:
            # Используем symusic.Score вместо miditoolkit.MidiFile (рекомендуется для miditok v3.0+)
            midi = Score(str(path))
            
            # Проверяем, что файл не пустой
            if len(midi.tracks) == 0:
                print(f"Warning: {path.name} has no tracks")
                continue
            
            # Проверяем количество нот
            total_notes = sum(len(track.notes) for track in midi.tracks)
            if total_notes == 0:
                print(f"Warning: {path.name} has no notes")
                continue
            
            # Токенизация - используем правильный метод
            # В miditok v3.0+ tokenizer() возвращает список TokSequence объектов
            try:
                tokens_seq = tokenizer(midi)
                
                # tokenizer(midi) возвращает список TokSequence объектов
                # Каждый TokSequence имеет атрибут ids со списком ID токенов
                tokens = []
                if isinstance(tokens_seq, list):
                    # Объединяем все ids из всех TokSequence в один список
                    for seq in tokens_seq:
                        if hasattr(seq, 'ids'):
                            ids = seq.ids
                            if isinstance(ids, list):
                                tokens.extend(ids)
                            elif hasattr(ids, '__iter__'):
                                tokens.extend(list(ids))
                elif hasattr(tokens_seq, 'ids'):
                    # Если это один TokSequence объект
                    ids = tokens_seq.ids
                    if isinstance(ids, list):
                        tokens = ids
                    elif hasattr(ids, '__iter__'):
                        tokens = list(ids)
                else:
                    tokens = []
                        
            except Exception as e:
                print(f"Error tokenizing {path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            token_lengths.append(len(tokens))
            
            if len(tokens) >= min_seq_len:
                all_tokens.append(tokens)
            else:
                skipped_short += 1
            
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if token_lengths:
        print(f"Token sequence lengths - min: {min(token_lengths)}, max: {max(token_lengths)}, "
              f"mean: {sum(token_lengths)/len(token_lengths):.1f}")
    print(f"Loaded and tokenized {len(all_tokens)} sequences (skipped {skipped_short} short files)")
    return all_tokens


# ────────────────────────────────────────────────
#   Создание обучающих последовательностей (sliding window)
# ────────────────────────────────────────────────

def create_sequences(
    token_lists: List[List[int]],
    context_len: int = 512,             # 2025–2026 типичный размер контекста
    stride: int = 128,                  # перекрытие — сильно влияет на качество
) -> Tuple[np.ndarray, np.ndarray]:
    inputs = []
    targets = []
    skipped_short = 0
    
    for tokens in token_lists:
        if len(tokens) <= context_len:
            skipped_short += 1
            continue
        
        for i in range(0, len(tokens) - context_len, stride):
            chunk = tokens[i : i + context_len + 1]   # +1 для target
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
    
    if not inputs:
        max_len = max((len(tokens) for tokens in token_lists), default=0)
        raise ValueError(
            f"No sequences long enough for context_len={context_len}. "
            f"Maximum sequence length found: {max_len}. "
            f"Consider reducing context_len or increasing min_seq_len."
        )
    
    if skipped_short > 0:
        print(f"Warning: {skipped_short} sequences were too short (<={context_len}) and skipped")
    
    X = np.array(inputs, dtype=np.int64)
    Y = np.array(targets, dtype=np.int64)
    
    print(f"Created {len(X)} training examples (context={context_len}, stride={stride})")
    return X, Y


# ────────────────────────────────────────────────
#   Разделение на train / val (по файлам, а не по кускам!)
# ────────────────────────────────────────────────

def split_train_val(
    token_lists: List[List[int]],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[List[int]], List[List[int]]]:
    random.seed(seed)
    random.shuffle(token_lists)
    
    split_idx = int(len(token_lists) * (1 - val_ratio))
    train_files = token_lists[:split_idx]
    val_files   = token_lists[split_idx:]
    
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
    return train_files, val_files


# ────────────────────────────────────────────────
#   PyTorch Dataset
# ────────────────────────────────────────────────

class MusicTokenDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.from_numpy(sequences).long()
        self.targets   = torch.from_numpy(targets).long()
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ────────────────────────────────────────────────
#   Главная функция подготовки данных
# ────────────────────────────────────────────────

def prepare_music_data(
    midi_dir: str = "data/",
    context_len: int = 256,  # Уменьшено с 512 для работы с более короткими последовательностями
    stride: int = 64,         # Уменьшено пропорционально
    val_ratio: float = 0.12,
    max_files: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    tokenizer_path: Optional[str] = None,  # Путь для сохранения/загрузки токенизатора
    min_seq_len: int = 32,    # Минимальная длина последовательности для сохранения
) -> Tuple[DataLoader, DataLoader, REMI]:
    
    from pathlib import Path
    
    # REMI токенизатор не требует обучения (rule-based), но можно сохранить/загрузить конфигурацию
    if tokenizer_path and Path(tokenizer_path).exists():
        print(f"Loading tokenizer from {tokenizer_path}")
        try:
            tokenizer = REMI.from_file(tokenizer_path)
        except:
            print("Failed to load tokenizer, creating new one...")
            tokenizer = create_remi_tokenizer()
    else:
        tokenizer = create_remi_tokenizer()
        
        # Сохраняем конфигурацию токенизатора для последующего использования
        if tokenizer_path:
            Path(tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                tokenizer.save(tokenizer_path)
                print(f"Tokenizer config saved to {tokenizer_path}")
            except Exception as e:
                print(f"Warning: Could not save tokenizer: {e}")
    
    # 1. Токенизация всех файлов
    token_lists = load_and_tokenize_midis(
        midi_dir, tokenizer, max_files=max_files, min_seq_len=min_seq_len
    )
    
    # 2. Разделение по файлам (чтобы избежать leakage)
    train_tokens, val_tokens = split_train_val(token_lists, val_ratio=val_ratio)
    
    # 3. Создание последовательностей
    X_train, Y_train = create_sequences(train_tokens, context_len, stride)
    X_val,   Y_val   = create_sequences(val_tokens,   context_len, stride)
    
    # 4. Datasets
    train_ds = MusicTokenDataset(X_train, Y_train)
    val_ds   = MusicTokenDataset(X_val,   Y_val)
    
    # 5. DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    return train_loader, val_loader, tokenizer


# Пример использования
if __name__ == "__main__":
    train_loader, val_loader, tokenizer = prepare_music_data(
        midi_dir="data/lmd_clean/",     # или твоя папка
        context_len=512,
        stride=64,                      # меньший stride → больше данных
        batch_size=48,                  # подбери под видеокарту
    )
    
    # Посмотреть один батч
    x, y = next(iter(train_loader))
    print(x.shape, y.shape)             # → [bs, 512], [bs, 512]
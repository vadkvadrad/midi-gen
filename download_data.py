#!/usr/bin/env python3
"""
Скрипт для загрузки данных с Kaggle.
Использует kagglehub для загрузки датасета классической музыки в формате MIDI.
"""

import kagglehub
from pathlib import Path

def download_midi_data(dataset_name: str = "soumikrakshit/classical-music-midi", output_dir: str = "data/midi"):
    """
    Загружает MIDI датасет с Kaggle.
    
    Args:
        dataset_name: Имя датасета на Kaggle (username/dataset-name)
        output_dir: Директория для сохранения данных
    """
    print(f"Downloading dataset: {dataset_name}")
    print("This may take a while...")
    
    # Загрузка последней версии датасета
    path = kagglehub.dataset_download(dataset_name)
    
    print(f"Dataset downloaded to: {path}")
    
    # Создаем целевую директорию
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Копируем MIDI файлы в целевую директорию
    source_path = Path(path)
    midi_files = list(source_path.rglob("*.mid")) + list(source_path.rglob("*.midi"))
    
    print(f"Found {len(midi_files)} MIDI files")
    
    for midi_file in midi_files:
        # Сохраняем структуру подпапок
        relative_path = midi_file.relative_to(source_path)
        target_file = output_path / relative_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Копируем файл
        import shutil
        shutil.copy2(midi_file, target_file)
    
    print(f"MIDI files copied to: {output_dir}")
    print(f"Total files: {len(midi_files)}")
    
    return output_dir

if __name__ == "__main__":
    import sys
    
    # Можно передать кастомный датасет через аргументы
    dataset = sys.argv[1] if len(sys.argv) > 1 else "soumikrakshit/classical-music-midi"
    output = sys.argv[2] if len(sys.argv) > 2 else "data/midi"
    
    download_midi_data(dataset, output)


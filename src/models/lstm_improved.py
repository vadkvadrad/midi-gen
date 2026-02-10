# src/models/music_lstm.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MusicLSTM(nn.Module):
    """
    Улучшенная LSTM-модель для генерации музыки.
    - Multi-layer LSTM (2-3 слоя) для лучшей capacity.
    - Embedding для токенов.
    - Dropout и LayerNorm для стабильности.
    - Поддержка variable-length sequences (с padding/packing).
    """
    
    def __init__(
        self,
        vocab_size: int,              # Размер словаря из REMI (len(tokenizer))
        embed_size: int = 256,        # Размер embedding (больше = лучше, но медленнее)
        hidden_size: int = 512,       # Размер hidden state (память LSTM)
        num_layers: int = 3,          # Кол-во слоёв LSTM (2-3 оптимально)
        dropout: float = 0.3,         # Dropout для регуляризации
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding слой: токены → векторы
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)  # 0 = PAD если нужно
        
        # LSTM: основной блок
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,         # [batch, seq, feat]
            dropout=dropout,
            bidirectional=False,      # Unidirectional для генерации
        )
        
        # LayerNorm для стабильности (новинка в 2026 туториалах)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Выходной слой: hidden → logits (вероятности токенов)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Инициализация весов (улучшает convergence)
        self._init_weights()
    
    def _init_weights(self):
        # Xavier для embedding и fc
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(
        self,
        x: torch.Tensor,              # [batch, seq_len]
        lengths: torch.Tensor = None, # Длины последовательностей (для packing)
        hidden: tuple = None,         # (h0, c0) для stateful
    ) -> tuple:
        """
        Forward pass.
        - Если lengths: используем packing для efficiency (рекомендую для больших seq).
        - Возвращает logits [batch, seq_len, vocab] и hidden state.
        """
        embed = self.embedding(x)  # [batch, seq_len, embed_size]
        
        if lengths is not None:
            # Packing: игнорируем padding
            embed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        out, hidden = self.lstm(embed, hidden)  # out: [batch, seq_len, hidden_size]
        
        if lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
        
        out = self.layer_norm(out)  # Нормализация
        logits = self.fc(out)      # [batch, seq_len, vocab_size]
        
        return logits, hidden


# Пример использования (для теста)
if __name__ == "__main__":
    vocab_size = 300  # Пример из REMI
    model = MusicLSTM(vocab_size)
    print(model)
    
    # Тестовый input
    x = torch.randint(0, vocab_size, (4, 512))  # batch=4, seq=512
    logits, hidden = model(x)
    print(logits.shape)  # Должен быть [4, 512, 300]
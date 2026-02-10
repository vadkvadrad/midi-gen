# Инструкция по работе с Docker

## Требования

- Docker (версия 20.10+)
- Docker Compose (версия 1.29+)
- Для GPU: NVIDIA Docker (nvidia-docker2)

## Быстрый старт

1. **Соберите образ:**
   ```bash
   make build
   # или
   docker-compose build
   ```

2. **Загрузите данные:**
   ```bash
   make download-data
   ```

3. **Обучите модель:**
   ```bash
   make train
   ```

4. **Сгенерируйте музыку:**
   ```bash
   make generate
   ```

## Доступные команды

Все команды можно выполнять через `make`:

- `make build` - собрать Docker образ
- `make up` - запустить контейнер в фоне
- `make down` - остановить контейнер
- `make shell` - войти в контейнер (интерактивный режим)
- `make download-data` - загрузить данные с Kaggle
- `make train` - обучить модель
- `make generate` - сгенерировать музыку
- `make clean` - очистить данные и модели
- `make help` - показать справку

## Интерактивная работа

Для работы в интерактивном режиме:

```bash
make shell
```

Внутри контейнера вы можете выполнять любые команды Python:

```bash
python train.py
python generate.py --gen-length 1000
python download_data.py custom/dataset-name
```

## Структура volumes

Docker Compose монтирует следующие директории:

- `./data` → `/app/data` - данные (MIDI файлы)
- `./models` → `/app/models` - обученные модели и токенизаторы
- `./output` → `/app/output` - сгенерированные MIDI файлы
- `./src` → `/app/src` - исходный код (можно редактировать)
- `./config` → `/app/config` - конфигурация
- `~/.kaggle` → `/root/.kaggle` - Kaggle credentials (read-only)

Все изменения в этих директориях сохраняются на хосте.

## GPU поддержка

Если у вас есть NVIDIA GPU:

1. Установите [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Раскомментируйте секцию `deploy` в `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

3. Пересоберите образ:
   ```bash
   docker-compose build
   ```

4. Проверьте доступность GPU:
   ```bash
   docker-compose run --rm midi-gen python -c "import torch; print(torch.cuda.is_available())"
   ```

## Решение проблем

### Проблема с Kaggle API

Если возникают проблемы с загрузкой данных:

1. Убедитесь, что `~/.kaggle/kaggle.json` существует на хосте
2. Проверьте права доступа: `chmod 600 ~/.kaggle/kaggle.json`
3. Проверьте, что файл монтируется в контейнер:
   ```bash
   docker-compose run --rm midi-gen ls -la /root/.kaggle/
   ```

### Проблемы с памятью

Если контейнер падает из-за нехватки памяти:

1. Уменьшите `batch_size` в `config/config.yml`
2. Уменьшите `context_len` в конфигурации
3. Ограничьте количество файлов через `max_files` в `prepare_music_data()`

### Очистка

Для полной очистки (осторожно, удалит все данные и модели):

```bash
make clean
```

Или вручную:
```bash
rm -rf data/midi/* models/*.pth models/*.json output/*.mid
```

## Продвинутое использование

### Запуск с кастомными параметрами

```bash
docker-compose run --rm midi-gen python generate.py \
    --model models/music_lstm.pth \
    --tokenizer models/tokenizer.json \
    --output output/custom.mid \
    --gen-length 2000 \
    --temperature 0.9
```

### Просмотр логов TensorBoard

```bash
docker-compose run --rm -p 6006:6006 midi-gen \
    tensorboard --logdir=runs --host=0.0.0.0
```

Затем откройте `http://localhost:6006` в браузере.

### Множественные эксперименты

Вы можете запускать несколько контейнеров одновременно с разными именами:

```bash
docker-compose -p experiment1 run --rm midi-gen python train.py
docker-compose -p experiment2 run --rm midi-gen python train.py
```

## Оптимизация образа

Для уменьшения размера образа можно использовать multi-stage build. Текущий Dockerfile использует `python:3.11-slim`, что уже оптимизировано для размера.

Для еще большей оптимизации можно:
- Использовать Alpine Linux (но может быть проблемы с компиляцией некоторых пакетов)
- Удалить кэш pip после установки
- Использовать `--no-cache-dir` для pip (уже используется)


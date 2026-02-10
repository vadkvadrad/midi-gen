# Makefile для удобной работы с Docker

.PHONY: build up down shell train generate download-data clean help

# Сборка образа
build:
	docker-compose build

# Запуск контейнера в интерактивном режиме
up:
	docker-compose up -d

# Остановка контейнера
down:
	docker-compose down

# Вход в контейнер
shell:
	docker-compose exec midi-gen /bin/bash

# Загрузка данных
download-data:
	docker-compose run --rm midi-gen python download_data.py

# Обучение модели
train:
	docker-compose run --rm midi-gen python train.py

# Генерация музыки
generate:
	docker-compose run --rm midi-gen python generate.py

# Очистка данных и моделей (осторожно!)
clean:
	docker-compose run --rm midi-gen sh -c "rm -rf data/midi/* models/*.pth models/*.json output/*.mid"

# Показать помощь
help:
	@echo "Доступные команды:"
	@echo "  make build          - Собрать Docker образ"
	@echo "  make up             - Запустить контейнер"
	@echo "  make down           - Остановить контейнер"
	@echo "  make shell          - Войти в контейнер"
	@echo "  make download-data  - Загрузить данные с Kaggle"
	@echo "  make train          - Обучить модель"
	@echo "  make generate       - Сгенерировать музыку"
	@echo "  make clean          - Очистить данные и модели"
	@echo "  make help           - Показать эту справку"


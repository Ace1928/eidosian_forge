# Slim image for CLI or API use
FROM python:3.11-alpine as base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN apk add --no-cache build-base && \
    pip install --no-cache-dir -r requirements.txt && \
    apk del build-base

COPY . .

CMD ["python", "labs/tutorial_app.py"]

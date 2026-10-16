FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir tensorflow-cpu==2.15.0 flask==3.0.3 \
    gunicorn==22.0.0 pillow==10.3.0 numpy==1.26.4

COPY src/ src/
COPY templates/ templates/
COPY models/ models/

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "src.app:app"]

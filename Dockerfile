FROM python:3.11-slim

WORKDIR /app

# Install system deps for torch/PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU for CI layer caching; CUDA version installed via requirements
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_ID=stabilityai/stable-diffusion-3.5-large
ENV HF_TOKEN=""

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

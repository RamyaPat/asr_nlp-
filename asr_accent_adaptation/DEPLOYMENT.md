# 🚀 Deployment Guide: Accent-Adaptive ASR System

This guide covers multiple deployment options for the Accent-Adaptive ASR system, from local development to cloud production.

---

## Table of Contents

1. [Local Development Setup](#1-local-development-setup)
2. [Docker Deployment](#2-docker-deployment)
3. [Hugging Face Spaces](#3-hugging-face-spaces)
4. [AWS Deployment](#4-aws-deployment)
5. [Google Cloud Platform](#5-google-cloud-platform)
6. [REST API Service](#6-rest-api-service)
7. [Edge Deployment](#7-edge-deployment)
8. [Monitoring & Logging](#8-monitoring--logging)

---

## 1. Local Development Setup

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU support)
- 8GB RAM minimum (16GB recommended)
- 5GB disk space

### Installation

```bash
# Clone repository
git clone <repository-url>
cd asr_accent_adaptation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run demo
cd demo
python app.py
```

Access at: `http://localhost:7860`

### GPU Configuration

**NVIDIA CUDA**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**AMD ROCm**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

**Apple Silicon (M1/M2)**:
```bash
# MPS backend enabled by default in PyTorch 2.0+
pip install torch torchvision torchaudio
```

---

## 2. Docker Deployment

### Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV MODEL_PATH=/app/models/final_model
ENV BASE_MODEL=openai/whisper-small

CMD ["python3", "demo/app.py", "--server_name", "0.0.0.0"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  asr-demo:
    build: .
    ports:
      - "7860:7860"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/final_model
      - BASE_MODEL=openai/whisper-small
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Build and Run

```bash
# Build image
docker build -t asr-accent-adaptive .

# Run container
docker run -p 7860:7860 --gpus all asr-accent-adaptive

# Or use docker-compose
docker-compose up
```

### CPU-Only Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "demo/app.py", "--server_name", "0.0.0.0"]
```

---

## 3. Hugging Face Spaces

### Setup

1. Create new Space at https://huggingface.co/spaces
2. Choose **Gradio** as the SDK
3. Select **Hardware**: CPU Basic or T4 Small

### Required Files

**`app.py`** (in root):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from demo.app import create_demo

demo = create_demo(
    model_path='openai/whisper-small',  # Use base model initially
    base_model='openai/whisper-small'
)

if __name__ == '__main__':
    demo.launch()
```

**`requirements.txt`**:
```
torch>=2.0.0
transformers>=4.35.0
gradio>=4.0.0
torchaudio>=2.0.0
peft>=0.7.0
jiwer>=3.0.0
numpy>=1.24.0
librosa>=0.10.0
soundfile>=0.12.0
```

**`README.md`** (for Space):
```markdown
---
title: Accent-Adaptive ASR
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Accent-Adaptive ASR System

Automatic Speech Recognition optimized for underrepresented accents.
```

### Upload Model

Option 1 - Use base model (no upload needed)
Option 2 - Upload fine-tuned model to Hugging Face Hub:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load your fine-tuned model
model = WhisperForConditionalGeneration.from_pretrained('./models/final_model')
processor = WhisperProcessor.from_pretrained('./models/final_model')

# Push to Hub
model.push_to_hub('your-username/accent-adaptive-asr')
processor.push_to_hub('your-username/accent-adaptive-asr')
```

Then update `app.py`:
```python
demo = create_demo(
    model_path='your-username/accent-adaptive-asr',
    base_model='openai/whisper-small'
)
```

---

## 4. AWS Deployment

### Option A: EC2 Instance

**Instance Recommendation**: `g4dn.xlarge` (T4 GPU, $0.526/hr)

```bash
# Launch EC2 instance with Deep Learning AMI
aws ec2 run-instances \
    --image-id ami-0c2d3e23e757b5d84 \
    --instance-type g4dn.xlarge \
    --key-name your-key \
    --security-groups asr-demo-sg

# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Clone and setup
git clone <repository-url>
cd asr_accent_adaptation
pip install -r requirements.txt

# Run demo
cd demo
python app.py --server_name 0.0.0.0
```

**Security Group Rules**:
- Inbound: Port 7860 (TCP) from your IP
- Inbound: Port 22 (SSH) from your IP

### Option B: AWS Lambda + API Gateway

**Limitations**:
- 15 min timeout
- 10GB memory max
- Cold start latency

**Setup**:

1. Package dependencies:
```bash
pip install -t ./package -r requirements.txt
cd package
zip -r ../deployment.zip .
cd ..
zip -g deployment.zip src/*.py
```

2. Create Lambda function:
```python
import json
import base64
from src.inference import WhisperInference

inferencer = None

def lambda_handler(event, context):
    global inferencer

    if inferencer is None:
        inferencer = WhisperInference(
            model_path='openai/whisper-small',
            base_model='openai/whisper-small'
        )

    # Decode base64 audio
    audio_data = base64.b64decode(event['body'])

    # Process
    result = inferencer.transcribe(audio_data)

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

3. Configure API Gateway for REST endpoint

### Option C: Amazon SageMaker

```python
from sagemaker.pytorch import PyTorchModel

pytorch_model = PyTorchModel(
    model_data='s3://your-bucket/model.tar.gz',
    role='SageMakerRole',
    entry_point='inference.py',
    framework_version='2.0',
    py_version='py310'
)

predictor = pytorch_model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1
)
```

---

## 5. Google Cloud Platform

### Option A: Compute Engine

```bash
# Create GPU instance
gcloud compute instances create asr-demo \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release

# SSH and setup
gcloud compute ssh asr-demo --zone=us-central1-a

# Install and run
git clone <repository-url>
cd asr_accent_adaptation
pip install -r requirements.txt
cd demo
python app.py --server_name 0.0.0.0
```

### Option B: Cloud Run

**`Dockerfile.cloudrun`**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg libsndfile1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD python demo/app.py --server_name 0.0.0.0 --server_port 8080
```

**Deploy**:
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT-ID/asr-demo

# Deploy
gcloud run deploy asr-demo \
    --image gcr.io/PROJECT-ID/asr-demo \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2
```

---

## 6. REST API Service

### FastAPI Implementation

**`api/main.py`**:
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from inference import WhisperInference

app = FastAPI(title="Accent-Adaptive ASR API")

inferencer = WhisperInference(
    model_path='../models/final_model',
    base_model='openai/whisper-small'
)

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    audio_data = await audio.read()

    # Save temporarily and process
    temp_path = f"/tmp/{audio.filename}"
    with open(temp_path, 'wb') as f:
        f.write(audio_data)

    result = inferencer.transcribe(temp_path)

    return JSONResponse({
        'transcription': result['transcription'],
        'processing_time': result['processing_time'],
        'rtf': result['rtf']
    })

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Run**:
```bash
pip install fastapi uvicorn python-multipart
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Test**:
```bash
curl -X POST "http://localhost:8000/transcribe" \
    -F "audio=@sample.wav"
```

### Flask Implementation

**`api/flask_app.py`**:
```python
from flask import Flask, request, jsonify
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from inference import WhisperInference

app = Flask(__name__)

inferencer = WhisperInference(
    model_path='../models/final_model',
    base_model='openai/whisper-small'
)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio = request.files['audio']
    temp_path = f"/tmp/{audio.filename}"
    audio.save(temp_path)

    result = inferencer.transcribe(temp_path)

    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

---

## 7. Edge Deployment

### ONNX Optimization

```bash
# Export to ONNX
cd src
python -c "
from inference import WhisperInference
inf = WhisperInference('../models/final_model', use_onnx=True)
"
```

### TensorRT Optimization (NVIDIA Jetson)

```python
import torch
from torch2trt import torch2trt

# Convert model to TensorRT
model_trt = torch2trt(
    model,
    [example_input],
    fp16_mode=True,
    max_batch_size=1
)

# Save
torch.save(model_trt.state_dict(), 'model_trt.pth')
```

### Raspberry Pi 4

**Optimizations**:
- Use Whisper-Tiny instead of Small
- Quantize to INT8
- Reduce batch size to 1
- Use ONNX Runtime with ARM optimizations

```bash
# Install ARM-optimized PyTorch
pip3 install torch torchvision torchaudio

# Install ONNX Runtime
pip3 install onnxruntime

# Run with optimizations
python inference.py \
    --model_path whisper-tiny \
    --device cpu \
    --use_onnx
```

---

## 8. Monitoring & Logging

### Logging Configuration

**`config/logging.yaml`**:
```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    formatter: standard
    filename: logs/asr.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  asr:
    level: INFO
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

transcription_counter = Counter('transcriptions_total', 'Total transcriptions')
processing_time = Histogram('processing_seconds', 'Processing time')
rtf_histogram = Histogram('rtf', 'Real-time factor')

@processing_time.time()
def transcribe_with_metrics(audio):
    result = inferencer.transcribe(audio)
    transcription_counter.inc()
    rtf_histogram.observe(result['rtf'])
    return result

# Start metrics server
start_http_server(9090)
```

### CloudWatch Integration (AWS)

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def log_metrics(result):
    cloudwatch.put_metric_data(
        Namespace='ASR',
        MetricData=[
            {
                'MetricName': 'ProcessingTime',
                'Value': result['processing_time'],
                'Unit': 'Seconds'
            },
            {
                'MetricName': 'RTF',
                'Value': result['rtf'],
                'Unit': 'None'
            }
        ]
    )
```

---

## Performance Recommendations

### Production Checklist

- [ ] Enable ONNX runtime for 27% latency reduction
- [ ] Use FP16 inference on GPU
- [ ] Implement request batching for throughput
- [ ] Set up load balancing for multiple instances
- [ ] Configure auto-scaling based on CPU/GPU usage
- [ ] Implement caching for repeated requests
- [ ] Set up monitoring and alerting
- [ ] Configure proper error handling and retries
- [ ] Use CDN for model file serving
- [ ] Implement rate limiting

### Cost Optimization

| Deployment | Cost/Hour | Use Case |
|------------|-----------|----------|
| Local GPU | $0 | Development |
| AWS g4dn.xlarge | $0.526 | Production |
| GCP T4 | $0.35 | Production |
| HF Spaces (T4) | $0.60 | Demo/Testing |
| AWS Lambda | Pay-per-use | Sporadic usage |

### Scaling Guidelines

**Single Instance**:
- Handles ~24 concurrent requests (RTF 0.42)
- Throughput: ~57 transcriptions/min

**Horizontal Scaling**:
- Use load balancer (ALB, NGINX)
- Deploy multiple instances
- Share model storage (S3, GCS)

**Vertical Scaling**:
- Better GPU: RTX 3090, A10, A100
- More RAM: Larger batch sizes
- Faster CPU: Better preprocessing

---

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce batch size
export BATCH_SIZE=1

# Use FP16
export USE_FP16=true
```

**Slow Inference on CPU**:
```bash
# Use ONNX
python inference.py --use_onnx

# Use Whisper-Tiny
python inference.py --model_name openai/whisper-tiny
```

**Docker GPU Not Found**:
```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## Security Considerations

1. **API Authentication**: Implement API keys or OAuth
2. **Rate Limiting**: Prevent abuse
3. **Input Validation**: Sanitize audio inputs
4. **HTTPS**: Use TLS in production
5. **Model Security**: Protect model weights
6. **Logging**: Don't log sensitive audio content
7. **Updates**: Keep dependencies patched

---

## Conclusion

This guide covers deployment from development to production across multiple platforms. Choose the option that best fits your:
- Budget
- Scalability needs
- Latency requirements
- Infrastructure expertise

For most use cases, we recommend:
- **Development**: Local setup
- **Demo**: Hugging Face Spaces
- **Production**: AWS EC2 or GCP Compute Engine with Docker
- **Enterprise**: Kubernetes cluster with auto-scaling

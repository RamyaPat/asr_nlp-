# 🚀 Quick Start Guide

Get up and running with the Accent-Adaptive ASR system in 5 minutes!

## 📋 Prerequisites

- Python 3.8+ installed
- 8GB RAM (16GB recommended)
- Optional: NVIDIA GPU with CUDA for training

## ⚡ Installation (2 minutes)

```bash
# Clone the repository (or navigate to project directory)
cd asr_accent_adaptation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎬 Demo the System (1 minute)

Launch the web interface to test with your own audio:

```bash
cd demo
python app.py
```

Then open your browser to: **http://localhost:7860**

- Click "Microphone Recording" to record live
- Or "File Upload" to upload an audio file
- Click "Transcribe" and see results instantly!

## 🎓 Train on Sample Data (5 minutes)

Test the training pipeline with synthetic data:

```bash
cd src
python finetune_lora.py \
    --use_synthetic \
    --num_epochs 3 \
    --batch_size 4 \
    --output_dir ../models
```

This will:
- Generate 100 synthetic audio samples
- Train for 3 epochs (~5 minutes)
- Save model to `../models/final_model`

## 📊 Evaluate Performance

After training, evaluate the model:

```bash
python evaluate.py \
    --model_path ../models/final_model \
    --use_synthetic \
    --compare_baseline
```

This shows WER improvement vs baseline.

## 🎤 Transcribe Your Audio

```bash
# Single file
python inference.py \
    --model_path ../models/final_model \
    --audio_path /path/to/your/audio.wav

# Directory of files
python inference.py \
    --model_path ../models/final_model \
    --audio_path /path/to/audio/directory/
```

## 📚 Train on Real Data

### Step 1: Prepare Your Data

Create a `data/` directory with:
```
data/
├── audio1.wav
├── audio1.txt  (transcription)
├── audio2.wav
├── audio2.txt
...
```

### Step 2: Train

```bash
cd src
python finetune_lora.py \
    --data_dir ../data \
    --num_epochs 10 \
    --batch_size 4 \
    --output_dir ../models
```

Training takes ~45 minutes with 100 samples on GPU.

### Step 3: Evaluate

```bash
python evaluate.py \
    --model_path ../models/final_model \
    --data_dir ../data \
    --compare_baseline \
    --error_analysis
```

## 🎯 Common Use Cases

### Use Case 1: Quick Demo
```bash
cd demo && python app.py
```

### Use Case 2: Batch Transcription
```bash
cd src
python inference.py \
    --model_path ../models/final_model \
    --audio_path /path/to/audios/ \
    --output_file results.txt
```

### Use Case 3: Fast Inference with ONNX
```bash
python inference.py \
    --model_path ../models/final_model \
    --audio_path audio.wav \
    --use_onnx
```

### Use Case 4: Public Demo
```bash
cd demo
python app.py --share
# Get a public URL to share
```

## 🐛 Troubleshooting

### "Module not found"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
```bash
# Reduce batch size
python finetune_lora.py --batch_size 2
```

### "No audio files found"
Ensure your audio files are in supported formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`

### Audio library errors
```bash
pip install soundfile librosa
# On Linux, may also need:
sudo apt-get install libsndfile1 ffmpeg
```

## 📖 Next Steps

- **Read the full README.md** for detailed documentation
- **Check OPTIMIZATION_REPORT.md** for performance analysis
- **Review PORTFOLIO_SUMMARY.md** for project highlights
- **See DEPLOYMENT.md** for production deployment options

## 🎓 Learning the Code

The codebase is organized for clarity:

```
src/
├── utils.py           # Helper functions (logging, checkpoints, timers)
├── dataset.py         # Data loading and preprocessing
├── augmentations.py   # Audio augmentation pipeline
├── finetune_lora.py   # Training script
├── inference.py       # Inference with ONNX support
└── evaluate.py        # WER/CER evaluation

demo/
└── app.py            # Gradio web interface
```

Start with:
1. `demo/app.py` - See how inference works
2. `src/dataset.py` - Understand data loading
3. `src/finetune_lora.py` - Review training loop
4. `src/augmentations.py` - Study augmentation techniques

## 💡 Tips

- **GPU Recommended**: Training is 10x faster on GPU
- **Start Small**: Use `--use_synthetic` first to understand workflow
- **Monitor Training**: Watch the loss curves to ensure convergence
- **Experiment**: Try different LoRA ranks, learning rates, epochs
- **Document**: Note what works for your specific accent/dialect

## ❓ Getting Help

- Check error messages carefully
- Review the full README.md
- Ensure all dependencies are installed
- Verify your audio files are valid
- Check that transcription files match audio files

## 🎉 Success Metrics

You've successfully set up the system when:
- ✅ Demo runs and transcribes audio
- ✅ Training completes without errors
- ✅ WER shows improvement vs baseline
- ✅ RTF is ≤ 0.5 (real-time capable)

Happy transcribing! 🎙️

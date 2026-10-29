# Data Directory

## Expected Data Format

Place your audio files and corresponding transcriptions in this directory.

### File Structure

```
data/
├── audio1.wav
├── audio1.txt
├── audio2.wav
├── audio2.txt
...
```

Each audio file should have a corresponding `.txt` file with the same name containing the transcription.

### Audio Requirements

- **Format**: WAV, MP3, FLAC, OGG, or M4A
- **Sample Rate**: Any (will be resampled to 16kHz automatically)
- **Channels**: Mono or Stereo (stereo will be converted to mono)
- **Duration**: Up to 30 seconds per sample (longer clips will be truncated)

### Transcription Format

- **Encoding**: UTF-8
- **Content**: Plain text transcription of the audio
- **Language**: English (or specify with `--language` flag)

### Example

**audio1.wav**: 3-second audio clip
**audio1.txt**:
```
The quick brown fox jumps over the lazy dog
```

## Dataset Splits

The system will automatically create train/validation/test splits:
- **Train**: 70%
- **Validation**: 15%
- **Test**: 15%

Generated manifest files:
- `train.json`
- `validation.json`
- `test.json`

## Sample Dataset

For testing purposes, you can use the synthetic dataset:

```bash
cd src
python finetune_lora.py --use_synthetic
```

This generates synthetic audio and transcriptions for rapid testing.

## Public Datasets

You can use these public datasets for accent adaptation:

### 1. Common Voice (Mozilla)
- **URL**: https://commonvoice.mozilla.org/
- **Accents**: Multiple languages and dialects
- **License**: CC0

### 2. VoxCeleb
- **URL**: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- **Content**: Speaker recognition dataset
- **Accents**: Various English accents

### 3. TIMIT
- **URL**: https://catalog.ldc.upenn.edu/LDC93S1
- **Content**: Phonetic transcriptions
- **Accents**: 8 major American English dialects

### 4. Speech Accent Archive
- **URL**: https://accent.gmu.edu/
- **Content**: English speakers from various native languages
- **Format**: WAV + IPA transcriptions

## Preparing Your Own Dataset

### Recording Guidelines

1. **Quality**: Use good microphone, quiet environment
2. **Duration**: 3-10 seconds per clip ideal
3. **Diversity**: Multiple speakers, contexts
4. **Quantity**: Minimum 50 samples, 200+ recommended

### Transcription Tips

1. **Accuracy**: Exact words spoken
2. **Punctuation**: Basic punctuation helps
3. **Capitalization**: Standard capitalization
4. **No special markers**: Avoid [laughs], [pause], etc.

### Data Augmentation

The training pipeline includes automatic augmentation:
- Speed perturbation (0.9x-1.1x)
- Pitch shift (±2 semitones)
- Background noise
- Volume changes
- Room reverb
- SpecAugment

So you don't need to manually augment your data.

## Troubleshooting

**Issue**: "No data found"
**Solution**: Ensure audio and txt files have matching names

**Issue**: "Audio loading error"
**Solution**: Install required libraries: `pip install soundfile librosa`

**Issue**: "Transcription encoding error"
**Solution**: Save txt files as UTF-8 encoding

## Data Privacy

- Never commit actual audio or transcription files to git
- Use `.gitignore` to exclude data files
- Consider encryption for sensitive data
- Follow data protection regulations (GDPR, CCPA, etc.)

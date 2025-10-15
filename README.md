# Toxicity Detection System - Natural Language Processing

A deep learning-based system that analyzes text for different types of toxicity using the state-of-the-art BERT model fine-tuned for toxicity detection.

## Overview 📋

This project implements a toxicity detection system that can analyze text input for various forms of toxic content. It utilizes the `unitary/toxic-bert` model, which is specifically fine-tuned for detecting different categories of toxic language.

## Features ✨

- Real-time toxicity analysis
- Multi-label classification for 6 different toxicity types:
  - General Toxicity
  - Severe Toxicity
  - Obscenity
  - Threat
  - Insult
  - Identity Hate
- Visual warning indicators for high toxicity scores
- Interactive command-line interface
- Sorted display of toxicity scores
- Support for both GPU and CPU inference

## Requirements 📦

```
python >= 3.7
torch
transformers
```

## Installation 🔧

1. Clone the repository:
```bash
git clone https://github.com/nirantbendale/NLP-Toxicity-Detection-System.git
cd toxicity-detector
```

2. Install the required packages:
```bash
pip install torch transformers
```

## Usage 🚀

Run the script:
```bash
python toxicity_detector.py
```

Enter text when prompted to receive toxicity analysis. Type 'quit' to exit the program.

### Example Usage:
```python
Enter a text to analyze (or 'quit' to exit):
This is a wonderful contribution!

Analysis Results:
--------------------------------------------------
   toxic: 0.012
   severe_toxic: 0.003
   obscene: 0.005
   threat: 0.001
   insult: 0.004
   identity_hate: 0.002
```

## Technical Details 🔍

### Model Architecture
- Base Model: BERT (Bidirectional Encoder Representations from Transformers)
- Fine-tuned Version: unitary/toxic-bert
- Output: Multi-label classification with sigmoid activation

### Libraries Used
- **PyTorch**: Deep learning framework for model operations
- **Transformers**: Hugging Face's transformers library for BERT model and tokenizer

### Implementation Details
1. **Text Preprocessing**:
   - Tokenization using BERT tokenizer
   - Padding and truncation to 128 tokens
   - Batch processing capability

2. **Model Inference**:
   - Automatic device selection (GPU/CPU)
   - Efficient batched predictions
   - Probability scores using sigmoid activation

3. **Output Processing**:
   - Sorted toxicity scores
   - Visual indicators for high toxicity
   - Formatted display of results

## Performance Considerations ⚡

- GPU acceleration when available
- Efficient batch processing
- Optimized inference pipeline
- Memory-efficient predictions

## Limitations 📝

- Maximum input length of 128 tokens
- Model performance depends on training data distribution
- May require fine-tuning for specific use cases
- English language focus

## Future Improvements 🔮

1. Add support for:
   - Multiple languages
   - Custom toxicity thresholds
   - Batch file processing
   - API endpoint

2. Implement:
   - Web interface
   - Detailed analysis reports
   - Custom model fine-tuning
   - Confidence scores


## Acknowledgments 🙏

- Hugging Face for the transformers library
- Unitary for the toxic-bert model
- PyTorch team for the deep learning framework

Rishi Kulkarni


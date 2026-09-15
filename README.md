# MedGemma Fine-tuning for Medical Chatbot

## About

This repository demonstrates a complete practical guide to fine-tuning Google's **MedGemma 27B** language model for building an empathetic medical assistant. The project uses **QLoRA** (4-bit quantization + Low-Rank Adaptation) to efficiently train a 27-billion-parameter model on consumer-grade GPUs.

This is ideal for researchers, ML engineers, and healthcare professionals exploring practical LLM fine-tuning with limited computational resources.

---

## Overview

This repository/notebook demonstrates the complete end-to-end process of fine-tuning the **google/medgemma-27b-text-it** Large Language Model to function as a warm, empathetic medical assistant.

## Key Features & Techniques

### Parameter-Efficient Fine-Tuning (PEFT)

Utilizes **QLoRA** (4-bit quantization combined with Low-Rank Adaptation) to make training a massive **27-billion-parameter model** feasible and memory-efficient on standard GPUs.

### Hugging Face Ecosystem

Built using modern ML libraries, including:

- [Transformers](https://huggingface.co/docs/transformers/)
- [PEFT](https://huggingface.co/docs/peft/)
- [TRL](https://huggingface.co/docs/trl/)
- [Datasets](https://huggingface.co/docs/datasets/)

### Conversational Formatting

Implements a custom prompt formatting pipeline that structures medical conversation data into:

- System instructions
- User queries
- Model responses

The formatted conversations follow the instruction-tuned format expected by the Gemma model.

### End-to-End Pipeline

The project covers the complete fine-tuning workflow:

1. Loading and formatting the medical conversation dataset.
2. Configuring the model for **4-bit precision**.
3. Attaching **LoRA adapters** for parameter-efficient training.
4. Performing **Supervised Fine-Tuning (SFT)** using a paged optimizer.
5. Running inference with the trained LoRA adapters.
6. Generating warm and empathetic medical responses.

## Workflow

```
Medical Conversation Dataset
             |
             v
  Data Loading & Formatting
             |
             v
   4-bit Model Quantization
             |
             v
      LoRA Adapters
             |
             v
  Supervised Fine-Tuning
             |
             v
   Trained LoRA Adapter
             |
             v
       Inference
             |
             v
Empathetic Medical Assistant
```

## Repository Structure

```
finetuning-medgemma-27b/
├── README.md                     # This file
├── finetuning_LoRA.ipynb         # Main fine-tuning notebook
└── finetuning_LoRA-2.ipynb       # Alternative version (same content)
```

## Getting Started

### Requirements

- GPU with sufficient VRAM (24GB+ recommended)
- Python 3.8+
- Hugging Face account (for model access)

### Installation

```bash
# Clone the repository
git clone https://github.com/Rishi-1482/finetuning-medgemma-27b.git
cd finetuning-medgemma-27b

# Install dependencies
pip install torch transformers peft trl datasets accelerate bitsandbytes
```

### Usage

1. Open `finetuning_LoRA.ipynb` in Jupyter Notebook or Google Colab
2. Follow the cells sequentially to:
   - Load the MedGemma model
   - Prepare your medical conversation dataset
   - Fine-tune with LoRA adapters
   - Run inference with your trained model

## Why This Approach?

- **Large Model (27B parameters)**: QLoRA 4-bit quantization reduces memory requirements
- **Limited GPU Memory**: Low-Rank Adapters (LoRA) make training feasible on consumer GPUs
- **Long Training Times**: Paged optimizer and gradient checkpointing accelerate training
- **Complex Setup**: Hugging Face TRL trainer provides production-ready implementation

## Model Information

- **Base Model**: google/medgemma-27b-text-it
- **Parameters**: 27 Billion
- **Type**: Instruction-tuned medical language model
- **Training Method**: Supervised Fine-Tuning (SFT) with LoRA

## Learning Resources

- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face TRL](https://huggingface.co/docs/trl/)
- [MedGemma Model Card](https://huggingface.co/google/medgemma-27b-text-it)

## Notes

- Both notebook files contain the same code and are provided as alternatives
- Adjust hyperparameters and batch sizes based on your available GPU memory
- Fine-tuning duration depends on dataset size and available computational resources

## License

This project is open source. Feel free to use and modify for your research and projects.

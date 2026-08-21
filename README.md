# MedGemma Fine-tuning for Medical Chatbot

## Overview

This repository/notebook demonstrates the complete end-to-end process of fine-tuning the **`google/medgemma-27b-text-it`** Large Language Model to function as a warm, empathetic medical assistant (General Practitioner).

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

```text
Medical Conversation Dataset
            │
            ▼
  Data Loading & Formatting
            │
            ▼
    4-bit Model Quantization
            │
            ▼
       LoRA Adapters
            │
            ▼
   Supervised Fine-Tuning
            │
            ▼
     Trained LoRA Adapter
            │
            ▼
         Inference
            │
            ▼
 Empathetic Medical Assistant

# Automated Product Description Generation From Images

This project focuses on the automated generation of concise and professional product titles for e-commerce marketplaces using state-of-the-art Vision-Language Models (VLM).

## 📂 Project Structure

The repository is organized into three main notebooks:
1.  **data.ipynb**: Data collection, filtering, and preprocessing.
2.  **model.ipynb**: Model configuration and fine-tuning process using LoRA.
3.  **evaluation.ipynb**: Model testing, inference, and performance comparison.

## 📊 Datasets

The final training set consists of **10,000 curated items** combined from the following sources:
*   **Electronics**: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
*   **Apparel & Accessories**: [Fashion Product Images Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

The consolidated and cleaned dataset used for this project is available on Hugging Face:
🔗 **[sdmikhalin/ecommerce-10k](https://huggingface.co/datasets/sdmikhalin/ecommerce-10k)**

## 🤖 Model Architecture

*   **Base Model**: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
*   **Training Method**: Parameter-efficient fine-tuning using **LoRA (Low-Rank Adaptation)**.

The model was trained to generate marketplace-ready titles (Brand, Gender/Audience, Color/Material, Type) while staying under a 12-word limit.

## 📈 Evaluation Results

The model was tested on 500 samples from the test set. The fine-tuned LoRA adapter significantly outperforms the Zero-shot base model across all language metrics.

### Overall Performance

| Metric | Zero-shot (Base) | Fine-tuned (LoRA) | Improvement |
| :--- | :---: | :---: | :---: |
| **BLEU-1** | 0.1001 | 0.3502 | **+249.9%** |
| **BLEU-4** | 0.0105 | 0.1097 | **+944.0%** |
| **ROUGE-L** | 0.3159 | 0.4238 | **+34.2%** |
| **BERTScore-F1** | 0.8362 | 0.8801 | **+5.3%** |

### Category-wise Highlights (BERTScore-F1)
*   **Accessories**: 0.8941 (+8.9%)
*   **Clothing**: 0.8889 (+6.2%)
*   **Electronics**: 0.8587 (+1.4%)

## 🚀 Live Demo

You can try the model prototype here:
🔗 **[E-commerce Description Generator Space](https://huggingface.co/spaces/sdmikhalin/new_space)**

---

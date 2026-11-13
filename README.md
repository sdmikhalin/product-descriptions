# Automated Product Description Generation From Images

This project focuses on the automated generation of concise and professional product titles for e-commerce marketplaces using state-of-the-art Vision-Language Models (VLM).

## 📂 Project Structure

The repository is organized into three main notebooks and two files:
1.  **dataset.ipynb**: Data collection, filtering, and preprocessing.
2.  **model.ipynb**: Model configuration and fine-tuning process using LoRA.
3.  **evaluation.ipynb**: Model testing, inference, and performance comparison.
4.  **comparison_results.csv**: File with the comparison results. It contains generated product descriptions for both models (basic and trained with LoRA) based on a test sample, including target names. 
5.  **comparison_examples.png**: an image with examples of generated descriptions for both models.

## 📊 Datasets

The final training set consists of **9,442 curated items** combined from the following sources:
*   **Electronics**: [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
*   **Apparel & Accessories**: [Fashion Product Images Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

The consolidated and cleaned dataset used for this project is available on Hugging Face:
🔗 **[sdmikhalin/ecommerce-10k](https://huggingface.co/datasets/sdmikhalin/ecommerce-10k)**

## 🤖 Model Architecture

*   **Base Model**: [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
*   **Training Method**: Parameter-efficient fine-tuning using **LoRA (Low-Rank Adaptation)**.

The model was trained to generate marketplace-ready titles (Brand, Gender/Audience, Color/Material, Type) while staying under a 12-word limit.

## 📈 Evaluation Results

The model was tested on 497 samples from the test set. The fine-tuned LoRA adapter significantly outperforms the Zero-shot base model across all language metrics.

### Overall Performance

| Metric | Zero-shot (Base) | Fine-tuned (LoRA) | Improvement |
| :--- | :---: | :---: | :---: |
| **BLEU-1** | 0.2639 | 0.3502 | **+32.7%** |
| **BLEU-4** | 0.0216 | 0.1097 | **+408.7%** |
| **ROUGE-L** | 0.3159 | 0.4238 | **+34.2%** |
| **BERTScore-F1** | 0.8655 | 0.8800 | **+1.7%** |

### Category-wise Highlights (BERTScore-F1)
*   **Accessories**: 0.8909 (+2.5%)
*   **Clothing**: 0.8907 (+2.1%)
*   **Electronics**: 0.8587 (+0.5%)



## 🚀 Live Demo

You can try the model prototype here:
🔗 **[E-commerce Description Generator Space](https://huggingface.co/spaces/sdmikhalin/new_space)**

## 🐳 Запуск в Docker

Код демо лежит в папке [`demo/`](demo/) (Gradio-приложение: Qwen2.5-VL-3B-Instruct + LoRA-адаптер
[`sdmikhalin/qwen25vl-3b-ecommerce-lora`](https://huggingface.co/sdmikhalin/qwen25vl-3b-ecommerce-lora)).
Веса модели в образ не входят: при первом запуске они скачиваются (~7.5 ГБ) в кэш Hugging Face
в volume `hf-cache` и при следующих запусках берутся оттуда.

### CPU (macOS / Linux, в т.ч. Apple Silicon)

```bash
docker build -t product-desc-demo demo
docker run --rm -p 7860:7860 -v hf-cache:/hf-cache product-desc-demo
```

Откройте http://localhost:7860, когда в логах появится `Running on local URL`.

- Модель на CPU грузится в `bfloat16` и занимает ~8 ГБ RAM. В Docker Desktop выделите контейнерам
  не меньше 10 ГБ памяти (Settings → Resources).
- Генерация одного описания на CPU занимает 1–2 минуты.
- Тип данных можно поменять так: `-e TORCH_DTYPE=float32` (нужно ~16 ГБ RAM).

### NVIDIA GPU (Linux-сервер)

На сервере нужны драйвер NVIDIA (с поддержкой CUDA 13.2) и
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build -f demo/Dockerfile.gpu -t product-desc-demo:gpu demo
docker run --rm --gpus all -p 7860:7860 -v hf-cache:/hf-cache product-desc-demo:gpu
```

### Полезное

- Чтобы использовать уже скачанный на хосте кэш, примонтируйте его вместо volume:
  `-v ~/.cache/huggingface:/hf-cache`.
- Удалить скачанные веса: `docker volume rm hf-cache`.
- Если первая загрузка весов идёт заметно медленнее, чем позволяет канал, отключите
  Xet-протокол Hugging Face и качайте по обычному HTTP: добавьте `-e HF_HUB_DISABLE_XET=1` в `docker run`.
- Пока модель скачивается и загружается, контейнер в статусе `health: starting`;
  `healthy` — значит, интерфейс готов.

---

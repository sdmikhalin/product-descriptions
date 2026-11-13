"""
E-commerce Product Description Generator
Powered by Qwen2.5-VL-3B + LoRA fine-tune
"""

import os

import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_ADAPTER = "sdmikhalin/qwen25vl-3b-ecommerce-lora"

MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

SYSTEM_PROMPT = (
    "You are an e-commerce product description assistant for an online marketplace.\n"
    "Generate concise product titles in this format: [Brand] [Gender/Audience] [Color/Material] [Type].\n"
    "Keep it under 12 words. Be specific. Avoid filler words like 'this', 'a', 'an', 'image of'."
)

CATEGORY_MAP = {
    "Auto": "auto",
    "Clothing": "clothing",
    "Electronics": "electronics",
    "Accessories": "accessories",
}

ALLOWED_CATEGORIES = ["clothing", "electronics", "accessories"]

USER_PROMPTS = {
    "clothing": "Describe this clothing item as a marketplace product title. Include brand, gender, color, and type.",
    "electronics": "Describe this electronic device as a marketplace product title. Include brand, model, color, and key specifications.",
    "accessories": "Describe this accessory as a marketplace product title. Include brand, gender, color, material, and type.",
}
DEFAULT_USER_PROMPT = "Describe this product as a concise marketplace title."


device = "cuda" if torch.cuda.is_available() else "cpu"
# TORCH_DTYPE overrides the default (e.g. "bfloat16" to halve RAM usage on CPU).
_dtype_name = os.environ.get("TORCH_DTYPE")
if _dtype_name:
    dtype = getattr(torch, _dtype_name)
else:
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"Loading model on {device} with dtype {dtype}", flush=True)

if device == "cpu" and dtype == torch.bfloat16:
    # PyTorch's CPU scaled_dot_product_attention kernel is extremely slow for
    # bfloat16 on some platforms (e.g. ~3.5 s per vision-encoder layer on arm64),
    # which made one request take ~20 min. Run attention in float32 instead;
    # weights stay in bfloat16, so memory usage is unchanged.
    _sdpa = torch.nn.functional.scaled_dot_product_attention

    def _sdpa_fp32(query, key, value, attn_mask=None, *args, **kwargs):
        if query.dtype == torch.bfloat16:
            if attn_mask is not None and attn_mask.is_floating_point():
                attn_mask = attn_mask.float()
            out = _sdpa(query.float(), key.float(), value.float(), attn_mask, *args, **kwargs)
            return out.to(query.dtype)
        return _sdpa(query, key, value, attn_mask, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = _sdpa_fp32

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    device_map=device,
)


model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()

processor = AutoProcessor.from_pretrained(
    LORA_ADAPTER,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

def build_user_prompt(category):
    return USER_PROMPTS.get(category, DEFAULT_USER_PROMPT)


def detect_category(image):
    image = image.convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text":
            "You are a product categorization assistant. "
            f"Choose ONE category from: {', '.join(ALLOWED_CATEGORIES)}. "
            "Reply with only the category word, nothing else."
        }]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"What category? Options: {', '.join(ALLOWED_CATEGORIES)}"},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False, num_beams=1)

    generated = output[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(generated, skip_special_tokens=True)[0].strip().lower()

    for cat in ALLOWED_CATEGORIES:
        if cat in answer:
            return cat
    return "clothing"


def generate_description(image, category, max_new_tokens=128):
    image = image.convert("RGB")
    user_prompt = build_user_prompt(category)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True).to(model.device)

    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            repetition_penalty=1.1,
            length_penalty=0.8,
            no_repeat_ngram_size=3,
        )

    generated = output[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def predict_caption(image, category_choice):
    if image is None:
        return "⚠️ Upload an image", "—"

    selected = CATEGORY_MAP[category_choice]

    if selected == "auto":
        category_en = detect_category(image)
        category_display = f"{category_en}"
    else:
        category_en = selected
        category_display = f"{category_en}"

    description = generate_description(image, category_en)
    return description, category_display


interface = gr.Interface(
    fn=predict_caption,
    inputs=[
        gr.Image(type="pil", label="Product photo", height=600),
        gr.Radio(
            choices=list(CATEGORY_MAP.keys()),
            value="Auto",
            label="Product category",
        ),
    ],
    outputs=[
        gr.Textbox(label="Description", lines=3),
        gr.Textbox(label="Category", lines=1),
    ],
    title="Product Description Generator",
    description=(
        "Upload a product photo and select a category. The model will generate a description for the marketplace."
        "Generation takes ~1-2 minutes"
    ),
)

interface.launch()

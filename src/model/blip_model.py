import torch.nn as nn
from transformers import BlipForConditionalGeneration, BlipProcessor

class BlipCaptioner(nn.Module):
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        super().__init__()
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)

    def forward(self, **batch):
        return self.model(**batch)

    def generate(self, pixel_values, max_length=32):
        return self.model.generate(pixel_values, max_length=max_length)
    

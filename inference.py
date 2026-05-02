import hydra
from omegaconf import DictConfig
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(cfg: DictConfig):
    processor = BlipProcessor.from_pretrained(cfg.paths.model_save_path)
    model = BlipForConditionalGeneration.from_pretrained(cfg.paths.model_save_path)

    # пример инференса
    image = Image.open("example.jpg").convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_length=32)
    print(processor.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
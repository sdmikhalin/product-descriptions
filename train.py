import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pandas as pd

from src.datasets.fashion_dataset import FashionDataset
from src.model.blip_model import BlipCaptioner
from src.trainer.trainer import BlipTrainer

@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(cfg: DictConfig):
    # загрузка данных
    df = pd.read_csv(cfg.data.metadata_path, error_bad_lines=False)
    df = df.sample(cfg.data.train_size + cfg.data.test_size, random_state=42)
    train_df = df.iloc[:cfg.data.train_size]
    val_df   = df.iloc[cfg.data.train_size:]

    # модель
    model = BlipCaptioner(cfg.model.name)

    # датасеты и загрузчики
    train_ds = FashionDataset(cfg.data.image_dir, train_df, model.processor)
    val_ds   = FashionDataset(cfg.data.image_dir, val_df, model.processor)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size,
                              shuffle=True, num_workers=cfg.data.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=cfg.data.batch_size,
                              shuffle=False, num_workers=cfg.data.num_workers)

    # обучение
    trainer = BlipTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.training.learning_rate,
        epochs=cfg.training.epochs,
        device=cfg.training.device
    )
    trained_model = trainer.fit()

    # сохранение
    trained_model.model.save_pretrained(cfg.paths.model_save_path)
    trained_model.processor.save_pretrained(cfg.paths.model_save_path)

if __name__ == "__main__":
    main()
import torch
import sys
import os

torch.set_default_tensor_type('torch.FloatTensor')

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from configs.voc_config import get_voc_config
from scripts.register_dataset import register_voc_datasets
from detectron2.engine import DefaultTrainer

if __name__ == "__main__":
    register_voc_datasets()

    cfg = get_voc_config()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("Training completed successfully.")

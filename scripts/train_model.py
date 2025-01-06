import sys
import os

# プロジェクトのルートディレクトリをモジュールパスに追加
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from configs.voc_config import get_voc_config
from scripts.register_dataset import register_voc_datasets
from detectron2.engine import DefaultTrainer

if __name__ == "__main__":
    # データセット登録
    register_voc_datasets()

    # 設定を取得
    cfg = get_voc_config()

    # トレーニング
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("Training completed successfully.")

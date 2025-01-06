from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_pascal_voc

def register_voc_datasets():
    if "voc_2012_train" not in DatasetCatalog.list():
        register_pascal_voc("voc_2012_train", "datasets/VOC2012", "train", year=2012)
    else:
        print("Dataset 'voc_2012_train' is already registered. Skipping...")

    if "voc_2012_val" not in DatasetCatalog.list():
        register_pascal_voc("voc_2012_val", "datasets/VOC2012", "val", year=2012)
    else:
        print("Dataset 'voc_2012_val' is already registered. Skipping...")

    print("VOC2012 datasets registered successfully.")

if __name__ == "__main__":
    register_voc_datasets()

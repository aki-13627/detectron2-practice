from detectron2.config import get_cfg

def get_voc_config():
    cfg = get_cfg()
    cfg.DATASETS.TRAIN = ("voc_2012_train",)
    cfg.DATASETS.TEST = ("voc_2012_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
    cfg.MODEL.DEVICE = "cpu"
    return cfg

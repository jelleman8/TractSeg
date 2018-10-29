import os
from tractseg.experiments.tract_seg import Config as HighResClassificationConfig


class Config(HighResClassificationConfig):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    DATA_AUGMENTATION = True
    DAUG_ELASTIC_DEFORM = True
    DAUG_ROTATE = True

    NUM_EPOCHS = 400
    MODEL = "UNet_Pytorch_weighted"
    CLASSES = "20_endpoints"
    LOSS_WEIGHT = 5
    LOSS_WEIGHT_LEN = -1
    TRAINING_SLICE_DIRECTION = "xyz"
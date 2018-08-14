import os
from tractseg.config.TractSegHP_LowRes import HP as TractSegHP_LowRes


class HP(TractSegHP_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    CLASSES = "20"
    NUM_EPOCHS = 300
    DATA_AUGMENTATION = False
    MODEL = "DAdapt_Model"
    LEARNING_RATE = 0.001   #0.001
    USE_VISLOGGER = True        # use ts_env_py3 env for that
    # WEIGHT_DECAY = 1e-5
    WEIGHT_DECAY = 0
    # ALPHA_LONGER = 8  # 4
    WARMUP_LEN = 20
    ALPHA_MAX = 0.001
    ALPHA_UPDATE_END = 100
    DOMAIN_LOSS_DIV = 40.
    INFO = "Discriminator DeepSup"
    BATCH_NORM = True
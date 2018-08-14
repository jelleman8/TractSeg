import os
from tractseg.config.TractSegHP_LowRes import HP as TractSegHP_LowRes


class HP(TractSegHP_LowRes):
    EXP_NAME = os.path.basename(__file__).split(".")[0]

    CLASSES = "20"
    NUM_EPOCHS = 200
    DATA_AUGMENTATION = False
    MODEL = "DAdapt_Model_BoNe"
    LEARNING_RATE = 0.001   #0.001
    USE_VISLOGGER = True        # use ts_env_py3 env for that
    WEIGHT_DECAY = 0    #1e-5
    INFO = "Discriminator BoNe; ALPHA_LONGER = 1"
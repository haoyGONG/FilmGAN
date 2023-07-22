import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR_SRC = "../dataset/dataset/train/src/"
TRAIN_DIR_REF = "../dataset/dataset/train/ref_small/"
VAL_DIR_SRC = "../dataset/data_crop/test/src/"
VAL_DIR_REF = "../dataset/data_crop/test/ref/"
BATCH_SIZE = 1
LEARNING_RATE = 1e-8
LAMBDA_STYLE = 1
LAMBDA_CYCLE = 1
LAMBDA_IDENTITY = 1
NUM_WORKERS = 1
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "./model/gen.pth.tar"
CHECKPOINT_CRITIC = "./model/critic.pth.tar"
CHECKPOINT_ENCODER = "./model/encoder.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)

transforms2 = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)
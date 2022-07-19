import math
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torchmetrics
from segmentation_models_pytorch import Unet


def convert_rgb_to_ids(labels: np.ndarray) -> np.ndarray:
  result = np.zeros(labels.shape[:2], dtype=np.uint8)
  result[np.where((labels == (0, 0, 255)).all(axis=2))] = 1
  result[np.where((labels == (0, 255, 0)).all(axis=2))] = 2
  result[np.where((labels == (255, 0, 0)).all(axis=2))] = 3
  return result


def convert_ids_to_rgb(labels: np.ndarray) -> np.ndarray:
  result = np.zeros((*labels.shape, 3), dtype=np.uint8)
  result[labels == 1] = (0, 0, 255)
  result[labels == 2] = (0, 255, 0)
  result[labels == 3] = (255, 0, 0)
  return result


class LunarDataset(torch.utils.data.Dataset):
    def __init__(self, path: Path, file_names: List[str], augment: bool = False):
        self._file_names = file_names
        self._images_dir = path / 'images'
        self._labels_dir = path / 'masks'
        self._augment = augment

        # self.image_size = (270, 480)
        self.image_size = (720, 1280)

        self.padded_image_size = (
            math.ceil(self.image_size[0] / 32) * 32,
            math.ceil(self.image_size[1] / 32) * 32
        )

        self.transforms = A.Compose([
            A.Resize(*self.image_size),
            A.PadIfNeeded(*self.padded_image_size),
            A.ToFloat(max_value=255),
            ToTensorV2()
        ])
        self.augmentations = A.Compose([
            A.Resize(*self.image_size),
            A.PadIfNeeded(*self.padded_image_size),

            A.HorizontalFlip(),

            A.ToFloat(max_value=255),
            ToTensorV2()
        ])

    def __getitem__(self, index: int):
        image_path = self._images_dir / self._file_names[index].replace('.png', '.jpg')
        labels_path = self._labels_dir / self._file_names[index]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = cv2.imread(str(labels_path))
        labels = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
        labels = convert_rgb_to_ids(labels)

        if self._augment:
            transformed = self.augmentations(image=image, mask=labels)
        else:
            transformed = self.transforms(image=image, mask=labels)

        return transformed['image'], transformed['mask'].type(torch.int64)

    def __len__(self):
        return len(self._file_names)


base_path = Path('LunarSeg/train')
train_names = sorted([path.name for path in (base_path / 'masks').iterdir()])

train_names, val_names = train_test_split(train_names, test_size=0.2, random_state=42)

train_dataset = LunarDataset(base_path, train_names, augment=True)
val_dataset = LunarDataset(base_path, val_names)




class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # self.network = Unet(encoder_name='resnet18', classes=4)
        # self.network = Unet(encoder_name='efficientnet-b0', encoder_weights='imagenet', classes=4)
        self.network = Unet(encoder_name='efficientnet-b6', encoder_weights='imagenet', classes=4)

        # self.loss_function = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.2, 0.2, 0.35, 0.25], dtype=np.float32)))
        self.loss_function = torch.nn.CrossEntropyLoss()

        metrics = torchmetrics.MetricCollection([
            torchmetrics.Precision(num_classes=4, average='macro', mdmc_average='samplewise'),
            torchmetrics.Recall(num_classes=4, average='macro', mdmc_average='samplewise'),
            torchmetrics.F1Score(num_classes=4, average='macro', mdmc_average='samplewise'),
            torchmetrics.Accuracy(num_classes=4, average='macro', mdmc_average='samplewise')
        ])
        self.train_metrics = metrics.clone('train_')
        self.val_metrics = metrics.clone('val_')

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, labels)
        self.log('train_loss', loss)

        outputs = torch.softmax(outputs, dim=1)
        self.log_dict(self.train_metrics(outputs, labels))

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)

        loss = self.loss_function(outputs, labels)
        self.log('val_loss', loss, prog_bar=True)

        outputs = torch.softmax(outputs, dim=1)
        self.log_dict(self.val_metrics(outputs, labels))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
        # return torch.optim.Adam(self.parameters(), lr=1e-3)


segmenter = Segmenter()

model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath='/checkpoints')
early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss', patience=10)
logger = pl.loggers.NeptuneLogger(
    api_key='YOUR_KEY',
    project='YOUR_PATH/LunarSeg'
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2)

trainer = pl.Trainer(logger=logger, callbacks=[model_checkpoint, early_stopping], gpus=1, max_epochs=100)
trainer.fit(segmenter, train_dataloaders=train_loader, val_dataloaders=val_loader)

logger.run.stop()


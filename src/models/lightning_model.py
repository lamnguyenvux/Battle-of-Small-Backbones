import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim import SGD, AdamW, Adam
from torchmetrics.classification import Accuracy
from typing import Literal

from src.models.backbones import get_model
from src.swats import SWATS
from src.utils import ComputeLoss, validate_literal_types


MODEL_ZOO = Literal[
    'wavemix', 'convnext-nano', 'efficientnetv2',
    'convnext-tiny', 'resnet', 'mobilenetv4', 'ghostnet', 'ghostnetv2',
    'regnet', 'shufflenet', 'repvgg-a0', 'repvgg-a1', 'repvgg-a2', 'repghostnet'
]

OPTIMIZER_TYPE = Literal['sgd', 'adam', 'adamw', 'swats', 'swatsw']


class LightningModel(LightningModule):
    def __init__(
        self,
        model_name: MODEL_ZOO,
        optimizer_type: OPTIMIZER_TYPE,
        num_classes: int = 2,
        l_rate: float = 0.001
    ):
        super().__init__()
        validate_literal_types(model_name, MODEL_ZOO)
        self.model_name = model_name
        self.optimizer_type = optimizer_type
        self.num_classes = num_classes
        self.l_rate = l_rate

        self.model = get_model(
            model_name=self.model_name,
            num_classes=self.num_classes
        )

        self.criterion = nn.CrossEntropyLoss()
        self.compute_loss = ComputeLoss()

        self.metric_accuracy: dict[str, dict[str, Accuracy]] = {}
        self.metric_accuracy['val'] = {}
        self.metric_accuracy['test'] = {}

        self.val_top1_acc = Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            top_k=1
        )

        self.test_top1_acc = Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            top_k=1
        )

        self.metric_accuracy['val']['top1'] = self.val_top1_acc
        self.metric_accuracy['test']['top1'] = self.test_top1_acc

        if self.num_classes >= 5:
            self.val_top5_acc = Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                top_k=5
            )
            self.test_top5_acc = Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                top_k=5
            )
            self.metric_accuracy['val']['top5'] = self.val_top5_acc
            self.metric_accuracy['test']['top5'] = self.test_top5_acc

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = self.criterion(logits, labels)
        self.compute_loss.update(loss)
        self.log(
            name="b_loss",
            value=loss,
            prog_bar=True,
            on_epoch=False,
            on_step=True
        )
        return loss

    def on_train_epoch_end(self):
        loss = self.compute_loss.compute()
        self.compute_loss.reset()

        self.log(
            name="loss",
            value=loss,
            on_epoch=True,
            prog_bar=True
        )

    def validation_step(self, batch, batch_idx):
        self.calculate_metric(batch, mode="val")

    def on_validation_epoch_end(self):
        self.calculate_metric_final("val")

    def test_step(self, batch, batch_idx):
        self.calculate_metric(batch, mode="test")

    def on_test_epoch_end(self):
        self.calculate_metric_final("test")

    def calculate_metric(self, batch, mode):
        images, labels = batch
        logits = self.forward(images)
        for topk, metric_acc in self.metric_accuracy[mode].items():
            metric_acc.update(logits, labels)

    def calculate_metric_final(self, mode: str):
        results = {}
        for topk, metric_acc in self.metric_accuracy[mode].items():
            results[mode + '_' + topk] = metric_acc.compute()
            metric_acc.reset()

        self.log_dict(
            dictionary=results,
            prog_bar=True,
            on_epoch=True
        )

    # def on_train_epoch_start(self):
    #     if self.current_epoch >= 10:
    #         self.trainer.accelerator.setup(self.trainer)

    def configure_optimizers(self):
        if self.optimizer_type == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.l_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
                amsgrad=False
            )
        elif self.optimizer_type == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.l_rate,
                momentum=0.9
            )
        elif self.optimizer_type == "adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=self.l_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                amsgrad=False
            )
        elif self.optimizer_type == "swats":
            optimizer = SWATS(
                self.model.parameters(),
                lr=self.l_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                amsgrad=False,
                nesterov=False,
                verbose=True
            )
        else:
            optimizer = SWATS(
                self.model.parameters(),
                lr=self.l_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                amsgrad=False,
                nesterov=False,
                decoupled_weight_decay=True,
                verbose=True
            )
        return optimizer


if __name__ == '__main__':
    import torch
    model_pl = LightningModel(
        model_name="convnext-nano",
        optimizer_type="sgd"
    )
    inp = torch.randn(2, 3, 224, 224)
    logits = model_pl(inp)
    print(logits)

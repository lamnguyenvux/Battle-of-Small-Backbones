import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim import SGD, AdamW
from torchmetrics.classification import Accuracy
from typing import Literal

from src.models.backbones import get_model
from src.utils import ComputeLoss, validate_literal_types


MODEL_ZOO = Literal[
    'wavemix', 'convnext-nano', 'efficientnetv2',
    'convnext-tiny', 'resnet', 'mobilenetv4', 'ghostnet',
    'regnet', 'shufflenet', 'repvgg-a0', 'repvgg-a1', 'repvgg-a2'
]


class LightningModel(LightningModule):
    def __init__(
        self, model_name: MODEL_ZOO,
        num_classes: int = 2,
        l_rate: float = 0.001
    ):
        super().__init__()
        validate_literal_types(model_name, MODEL_ZOO)
        self.model_name = model_name
        self.num_classes = num_classes
        self.l_rate = l_rate

        self.model = get_model(
            model_name=self.model_name,
            num_classes=self.num_classes
        )

        self.criterion = nn.CrossEntropyLoss()
        self.compute_loss = ComputeLoss()
        
        self.accuracy_metric = {}

        self.top1_acc = Accuracy(
            task="multiclass",
            num_classes=self.num_classes,
            top_k=1
        )

        self.accuracy_metric['top1'] = self.top1_acc

        if self.num_classes >= 5:
            self.top5_acc = Accuracy(
                task="multiclass",
                num_classes=self.num_classes,
                top_k=5
            )
            self.accuracy_metric['top5'] = self.top5_acc

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
        self.calculate_metric(batch)

    def on_validation_epoch_end(self):
        self.calculate_metric_final("val")

    def test_step(self, batch, batch_idx):
        self.calculate_metric(batch)
    
    def on_test_epoch_end(self):
        self.calculate_metric_final("test")

    def calculate_metric(self, batch):
        images, labels = batch
        logits = self.forward(images)
        for topk, metric_acc in self.accuracy_metric.items():
            metric_acc.update(logits, labels)

    def calculate_metric_final(self, mode: str):
        results = {}
        for topk, metric_acc in self.accuracy_metric.items():
            results[mode + '_' + topk] = metric_acc.compute()
            metric_acc.reset()

        self.log_dict(
            dictionary=results,
            prog_bar=True,
            on_epoch=True
        )

    def on_train_epoch_start(self):
        if self.current_epoch >= 10:
            self.trainer.accelerator.setup(self.trainer)

    def configure_optimizers(self):
        if self.current_epoch < 10:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.l_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
                amsgrad=False
            )
        else:
            optimizer = SGD(
                self.model.parameters(),
                lr=self.l_rate,
                momentum=0.9
            )
        return optimizer


if __name__ == '__main__':
    import torch
    model_pl = LightningModel(
        model_name="convnext-nano"
    )
    inp = torch.randn(2, 3, 224, 224)
    logits = model_pl(inp)
    print(logits)

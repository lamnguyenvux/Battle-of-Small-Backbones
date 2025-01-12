from src.models import LightningModel
from src.dataset import DataModule
import argparse
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="name of dataset: \
            cifar10, cifar100, cub200, dtd, \
            eurosat, flowers102, food101, platclef"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="path to root directory of dataset"
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        help="name of model: \
            'wavemix', 'convnext-nano', 'efficientnetv2', \
            'convnex-tiny', 'resnet', 'mobilenetv4', 'ghostnet', \
            'regnet', 'shufflenet', 'repvgg-a0', 'repvgg-a1', 'repvgg-a2'"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help="Batch size for training and validation"
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=2,
        help="Number of classes"
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=10,
        help="Number of epochs"
    )
    parser.add_argument(
        '--precision',
        type=str,
        default="32-true",
        help="precision training. Other options: Half precision (16, 16-mixed) | \
            Double precison (64, 64-true) | bfloat16 mixed precision (bf16, bf16-mixed) | \
            Full precision (32, 32-true)"
    )
    parser.add_argument(
        '--fast-dev-run',
        action="store_true",
        default=False,
        help="Quick test"
    )
    parser.add_argument(
        '--log-root-dir',
        type=str,
        default="/teamspace/studios/this_studio",
        help="log directory"
    )
    args = parser.parse_args()

    data = DataModule(
        dataset_type=args.dataset_name,
        root=args.root,
        batch_size=args.batch_size
    )

    model = LightningModel(
        model_name=args.model_name,
        num_classes=args.num_classes,
        l_rate=0.001
    )

    trainer = Trainer(
        precision=args.precision,
        max_epochs=args.max_epochs,
        fast_dev_run=args.fast_dev_run,
        default_root_dir=args.log_root_dir
    )
    trainer.fit(
        model=model,
        datamodule=data
    )

from lightning.pytorch.cli import LightningCLI

from src.models import LightningModel
from src.dataset import DataModule

def main():
    cli = LightningCLI(
        model_class=LightningModel,
        datamodule_class=DataModule,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == '__main__':
    main()
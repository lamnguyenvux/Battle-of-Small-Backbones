import torch
from torchmetrics import Metric
from typing import get_args


class ComputeLoss(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("num_loss", default=torch.tensor(
            0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(
            0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, loss_batch):
        self.num_loss += 1
        self.total += loss_batch

    def compute(self):
        return self.total / self.num_loss


def validate_literal_types(value, types):
    if value not in get_args(types):
        raise ValueError(
            f"{value} is not supported.\nSupported value are {get_args(types)}")

import torch
from torch._export.db.case import export_case, SupportLevel
from torch.utils import _pytree as pytree

    Pytree from PyTorch cannot be captured by TorchDynamo.
    
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override
Convert the wrapped batchnorm layers back to regular batchnorm layers.

        Args:
            model: Reference to the current LightningModule

        Return:
            LightningModule with regular batchnorm layers that will no longer sync across processes.

        
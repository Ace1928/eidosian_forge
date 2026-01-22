import torch
from torch.optim import Optimizer
from bitsandbytes.optim.optimizer import Optimizer1State
Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        
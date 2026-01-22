import inspect
import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch.nn.utils.prune as pytorch_prune
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor, nn
from typing_extensions import TypedDict, override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_debug, rank_zero_only
def make_pruning_permanent(self, module: nn.Module) -> None:
    """Removes pruning buffers from any pruned modules.

        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/nn/utils/prune.py#L1118-L1122

        """
    for _, module in module.named_modules():
        for k in list(module._forward_pre_hooks):
            hook = module._forward_pre_hooks[k]
            if isinstance(hook, pytorch_prune.BasePruningMethod):
                hook.remove(module)
                del module._forward_pre_hooks[k]
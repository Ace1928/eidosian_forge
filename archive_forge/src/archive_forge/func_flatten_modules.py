import logging
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union
import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
@staticmethod
def flatten_modules(modules: Union[Module, Iterable[Union[Module, Iterable]]]) -> List[Module]:
    """This function is used to flatten a module or an iterable of modules into a list of its leaf modules (modules
        with no children) and parent modules that have parameters directly themselves.

        Args:
            modules: A given module or an iterable of modules

        Returns:
            List of modules

        """
    if isinstance(modules, ModuleDict):
        modules = modules.values()
    if isinstance(modules, Iterable):
        _flatten_modules = []
        for m in modules:
            _flatten_modules.extend(BaseFinetuning.flatten_modules(m))
        _modules = iter(_flatten_modules)
    else:
        _modules = modules.modules()
    return [m for m in _modules if not list(m.children()) or m._parameters]
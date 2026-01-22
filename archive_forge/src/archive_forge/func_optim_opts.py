from typing import Dict, Any, Union, List, Tuple, Optional
from abc import ABC, abstractmethod
import random
import os
import torch
import parlai.utils.logging as logging
from torch import optim
from parlai.core.opt import Opt
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.nn.lr_scheduler import ParlAILRScheduler
from parlai.core.message import Message
from parlai.utils.distributed import is_distributed
from parlai.utils.misc import AttrDict, warn_once
from parlai.utils.fp16 import (
from parlai.core.metrics import (
from parlai.utils.distributed import is_primary_worker
from parlai.utils.torch import argsort, compute_grad_norm, padded_tensor, atomic_save
@classmethod
def optim_opts(self):
    """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
    optims = {k.lower(): v for k, v in optim.__dict__.items() if not k.startswith('__') and k[0].isupper()}
    try:
        import apex.optimizers.fused_adam as fused_adam
        import apex.optimizers.fused_lamb as fused_lamb
        optims['fused_adam'] = fused_adam.FusedAdam
        optims['fused_lamb'] = fused_lamb.FusedLAMB
    except ImportError:
        pass
    try:
        from qhoptim.pyt import QHM, QHAdam
        optims['qhm'] = QHM
        optims['qhadam'] = QHAdam
    except ImportError:
        pass
    optims['mem_eff_adam'] = MemoryEfficientFP16Adam
    optims['adafactor'] = Adafactor
    return optims
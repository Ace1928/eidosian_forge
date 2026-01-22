import logging
import warnings
from collections import OrderedDict
from typing import Union, Iterable, Dict
import torch
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.utils as utils

        Averages parameters or parameter groups of an optimizer if ``step`` is no less than ``warmup_steps``
        and it can be divided by a period in the keys of ``period_process_group_dict``,
        where ``step`` is increased by 1 at each iteration in the training loop.
        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        only the largest period is used, and the corresponding process group is used for averaging parameters.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.
        
import copy
import datetime
import io
import json
import math
import os
import sys
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import StreamHandler
from typing import Any, Dict, Iterator, List, Optional, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .tokenization_utils_base import BatchEncoding
from .utils import is_sagemaker_mp_enabled, is_torch_tpu_available, is_training_run_on_sagemaker, logging
def smp_gather(tensor):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)((smp_gather(t) for t in tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: smp_gather(v) for k, v in tensor.items()})
    elif not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Can't gather the values of type {type(tensor)}, only of nested list/tuple/dicts of tensors.")
    all_tensors = smp.allgather(tensor, smp.CommGroup.DP_GROUP)
    all_tensors = [atleast_1d(t) for t in all_tensors]
    return torch.cat([t.cpu() for t in all_tensors], dim=0)
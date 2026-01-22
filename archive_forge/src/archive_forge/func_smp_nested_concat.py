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
def smp_nested_concat(tensor):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)((smp_nested_concat(t) for t in tensor))
    elif isinstance(tensor, dict):
        return type(tensor)({k: smp_nested_concat(v) for k, v in tensor.items()})
    return tensor.concat().detach().cpu()
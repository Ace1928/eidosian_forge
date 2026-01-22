import copy
import itertools
import math
import os
import random
import sys
import tempfile
import time
from collections import namedtuple, OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import reduce
from typing import Union, NamedTuple, Callable, Any
import unittest
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import torch.distributed.algorithms.model_averaging.averagers as averagers
import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
import torch.distributed.algorithms.model_averaging.utils as model_averaging_utils
import torch.nn as nn
import torch.nn.functional as F
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.algorithms.ddp_comm_hooks import (
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.distributed.distributed_c10d import (
from torch.distributed.utils import (
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _dump_DDP_relevant_env_vars, _MixedPrecision
from torch.testing._internal.common_distributed import (
from torch.testing._internal.common_utils import (
import torch.distributed.optim.post_localSGD_optimizer as post_localSGD_optimizer
from torch.utils.data.distributed import DistributedSampler
@skip_but_pass_in_sandcastle_if(BACKEND not in DistTestCases.backend_feature['ddp'], f'The {BACKEND} backend does not support DistributedDataParallel')
@skip_if_lt_x_gpu(int(os.environ['WORLD_SIZE']))
def test_DistributedSampler_padding(self):
    world_size = dist.get_world_size()
    dataset_size = 100 + world_size + 1
    dataset = [torch.ones(1).to(self.rank) * i for i in range(dataset_size)]
    dataset_tiny_size = max(world_size // 2 - 1, 1)
    dataset_tiny = [torch.ones(1).to(self.rank) * i for i in range(dataset_tiny_size)]
    dist_sampler = DistributedSampler(dataset=dataset, drop_last=True)
    local_num_samples, local_dataset_size = (dist_sampler.num_samples, dist_sampler.total_size)
    effective_dataset_size = math.ceil((dataset_size - world_size) / world_size) if dataset_size % world_size != 0 else dataset_size / world_size
    self.assertEqual(local_num_samples, effective_dataset_size)
    self.assertEqual(local_dataset_size, local_num_samples * world_size)
    indices_list = list(iter(dist_sampler))
    self.assertEqual(len(indices_list), local_num_samples)

    def validate_global_samples(local_num_samples):
        world_samples = [torch.LongTensor([0]).to(self.rank) for _ in range(world_size)]
        dist.all_gather(world_samples, torch.tensor([local_num_samples]).to(self.rank))
        world_samples = [sample.item() for sample in world_samples]
        self.assertEqual(len(set(world_samples)), 1)
    validate_global_samples(local_num_samples)
    dist_sampler_added_samples = DistributedSampler(dataset=dataset)
    local_num_samples, local_dataset_size = (dist_sampler_added_samples.num_samples, dist_sampler_added_samples.total_size)
    self.assertEqual(local_num_samples, math.ceil(dataset_size / world_size))
    self.assertEqual(local_dataset_size, local_num_samples * world_size)
    indices_list = list(iter(dist_sampler_added_samples))
    self.assertEqual(len(indices_list), local_num_samples)
    validate_global_samples(local_num_samples)
    dist_sampler_added_samples_tiny = DistributedSampler(dataset=dataset_tiny)
    local_num_samples, local_dataset_size = (dist_sampler_added_samples_tiny.num_samples, dist_sampler_added_samples_tiny.total_size)
    self.assertEqual(local_num_samples, math.ceil(dataset_tiny_size / world_size))
    self.assertEqual(local_dataset_size, local_num_samples * world_size)
    indices_list = list(iter(dist_sampler_added_samples_tiny))
    self.assertEqual(len(indices_list), local_num_samples)
    validate_global_samples(local_num_samples)
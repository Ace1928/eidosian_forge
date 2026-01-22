import contextlib
import io
import math
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.data_loader import SeedableRandomSampler, prepare_data_loader
from accelerate.state import AcceleratorState
from accelerate.test_utils import RegressionDataset, are_the_same_tensors
from accelerate.utils import (
def rng_sync_check():
    state = AcceleratorState()
    synchronize_rng_states(['torch'])
    assert are_the_same_tensors(torch.get_rng_state()), 'RNG states improperly synchronized on CPU.'
    if state.distributed_type == DistributedType.MULTI_GPU:
        synchronize_rng_states(['cuda'])
        assert are_the_same_tensors(torch.cuda.get_rng_state()), 'RNG states improperly synchronized on GPU.'
    elif state.distributed_type == DistributedType.MULTI_XPU:
        synchronize_rng_states(['xpu'])
        assert are_the_same_tensors(torch.xpu.get_rng_state()), 'RNG states improperly synchronized on XPU.'
    generator = torch.Generator()
    synchronize_rng_states(['generator'], generator=generator)
    assert are_the_same_tensors(generator.get_state()), 'RNG states improperly synchronized in generator.'
    if state.local_process_index == 0:
        print('All rng are properly synched.')
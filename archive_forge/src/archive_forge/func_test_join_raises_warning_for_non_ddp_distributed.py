import warnings
from typing import List
from unittest.mock import Mock
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from accelerate.accelerator import Accelerator, DataLoaderConfiguration
from accelerate.utils.dataclasses import DistributedType
def test_join_raises_warning_for_non_ddp_distributed(accelerator):
    with warnings.catch_warnings(record=True) as w:
        with accelerator.join_uneven_inputs([Mock()]):
            pass
        assert issubclass(w[-1].category, UserWarning)
        assert 'only supported for multi-GPU' in str(w[-1].message)
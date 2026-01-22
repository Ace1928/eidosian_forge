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
def test_reinstantiated_state():
    import pytest
    AcceleratorState._reset_state()
    simple_model = torch.nn.Linear(1, 1)
    accelerator = Accelerator()
    AcceleratorState._reset_state()
    with pytest.raises(AttributeError) as cm:
        accelerator.prepare(simple_model)
    assert '`AcceleratorState` object has no attribute' in str(cm.value.args[0])
    assert 'This happens if `AcceleratorState._reset_state()`' in str(cm.value.args[0])
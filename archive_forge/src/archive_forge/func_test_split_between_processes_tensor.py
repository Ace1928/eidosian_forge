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
def test_split_between_processes_tensor():
    state = AcceleratorState()
    if state.num_processes > 1:
        data = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]).to(state.device)
        with state.split_between_processes(data) as results:
            if state.process_index == 0:
                assert torch.allclose(results, torch.tensor([0, 1, 2, 3]).to(state.device))
            else:
                assert torch.allclose(results, torch.tensor([4, 5, 6, 7]).to(state.device))
    state.wait_for_everyone()
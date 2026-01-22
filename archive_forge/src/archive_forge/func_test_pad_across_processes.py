import torch
from accelerate import PartialState
from accelerate.test_utils.testing import assert_exception
from accelerate.utils.dataclasses import DistributedType
from accelerate.utils.operations import (
def test_pad_across_processes(state):
    if state.is_main_process:
        tensor = torch.arange(state.num_processes + 1).to(state.device)
    else:
        tensor = torch.arange(state.num_processes).to(state.device)
    padded_tensor = pad_across_processes(tensor)
    assert padded_tensor.shape == torch.Size([state.num_processes + 1])
    if not state.is_main_process:
        assert padded_tensor.tolist() == list(range(0, state.num_processes)) + [0]
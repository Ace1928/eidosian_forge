import torch
from accelerate import PartialState
from accelerate.test_utils.testing import assert_exception
from accelerate.utils.dataclasses import DistributedType
from accelerate.utils.operations import (
def test_copy_tensor_to_devices(state):
    if state.distributed_type not in [DistributedType.MULTI_GPU, DistributedType.XLA]:
        return
    if state.is_main_process:
        tensor = torch.tensor([1, 2, 3], dtype=torch.int).to(state.device)
    else:
        tensor = None
    tensor = copy_tensor_to_devices(tensor)
    assert torch.allclose(tensor, torch.tensor([1, 2, 3], dtype=torch.int, device=state.device))
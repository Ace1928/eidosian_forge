import torch
from accelerate import PartialState
from accelerate.test_utils.testing import assert_exception
from accelerate.utils.dataclasses import DistributedType
from accelerate.utils.operations import (
def test_gather_non_contigous(state):
    if state.distributed_type == DistributedType.XLA:
        return
    tensor = torch.arange(12).view(4, 3).t().to(state.device)
    assert not tensor.is_contiguous()
    _ = gather(tensor)
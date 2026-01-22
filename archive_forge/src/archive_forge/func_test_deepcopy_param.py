import copy
import os
import pickle
from tempfile import TemporaryDirectory
import pytest
import torch
import bitsandbytes as bnb
from tests.helpers import TRUE_FALSE, torch_load_from_buffer, torch_save_to_buffer
def test_deepcopy_param():
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    param = bnb.nn.Params4bit(data=tensor, requires_grad=False).cuda(0)
    copy_param = copy.deepcopy(param)
    assert param.quant_state is not copy_param.quant_state
    assert param.data.data_ptr() != copy_param.data.data_ptr()
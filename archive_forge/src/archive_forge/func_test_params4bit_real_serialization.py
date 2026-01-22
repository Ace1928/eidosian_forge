import copy
import os
import pickle
from tempfile import TemporaryDirectory
import pytest
import torch
import bitsandbytes as bnb
from tests.helpers import TRUE_FALSE, torch_load_from_buffer, torch_save_to_buffer
def test_params4bit_real_serialization():
    original_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    original_param = bnb.nn.Params4bit(data=original_tensor, quant_type='fp4')
    original_param.cuda(0)
    serialized_param = pickle.dumps(original_param)
    deserialized_param = pickle.loads(serialized_param)
    assert torch.equal(original_param.data, deserialized_param.data)
    assert original_param.requires_grad == deserialized_param.requires_grad == False
    assert original_param.quant_type == deserialized_param.quant_type
    assert original_param.blocksize == deserialized_param.blocksize
    assert original_param.compress_statistics == deserialized_param.compress_statistics
    assert original_param.quant_state == deserialized_param.quant_state
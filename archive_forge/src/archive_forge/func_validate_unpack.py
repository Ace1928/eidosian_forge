from collections import OrderedDict
import gymnasium as gym
from typing import Union, Dict, List, Tuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
def validate_unpack(self, dnc_output, unpacked_state):
    """Ensure the unpacked state shapes match the DNC output"""
    s_ctrl_hidden, s_memory_dict, s_read_vecs = unpacked_state
    ctrl_hidden, memory_dict, read_vecs = dnc_output
    for i in range(len(ctrl_hidden)):
        for j in range(len(ctrl_hidden[i])):
            assert s_ctrl_hidden[i][j].shape == ctrl_hidden[i][j].shape, f'Controller state mismatch: got {s_ctrl_hidden[i][j].shape} should be {ctrl_hidden[i][j].shape}'
    for k in memory_dict:
        assert s_memory_dict[k].shape == memory_dict[k].shape, f'Memory state mismatch at key {k}: got {s_memory_dict[k].shape} should be {memory_dict[k].shape}'
    assert s_read_vecs.shape == read_vecs.shape, f'Read state mismatch: got {s_read_vecs.shape} should be {read_vecs.shape}'
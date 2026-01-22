import operator
import torch
from torch.ao.quantization.backend_config import (
from typing import List
def get_linear_configs():
    linear_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    linear_configs.append(BackendPatternConfig(torch.ops.aten.addmm.default).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_input_type_to_index({'weight': 2, 'bias': 0}))
    linear_configs.append(BackendPatternConfig(torch.ops.aten.mm.default).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_input_type_to_index({'weight': 1}))
    return linear_configs
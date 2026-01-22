import torch
import numpy as np
import argparse
from typing import Dict
def runDefaultTestWithSeed(seed):
    seed_tensor, tensor_list = prepareInputTensorsToRandomTopoTest(seed, MAX_TENSOR, MAX_TENSOR_DIM, MAX_TENSOR_SIZE, DEBUG_TENSOR, DEVICE, DTYPE)
    o = random_topology_test(seed_tensor, *tensor_list)
    traced_model = torch.jit.trace(random_topology_test, (seed_tensor, *tensor_list))
    jit_o = traced_model(seed_tensor, *tensor_list)
    jit_o = traced_model(seed_tensor, *tensor_list)
    validate_o = zip(o, jit_o)
    for oo, jit_oo in validate_o:
        if not oo.allclose(jit_oo, atol=1e-05, equal_nan=True):
            return False
    return True
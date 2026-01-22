import torch
import numpy as np
import argparse
from typing import Dict
def runTest(seed, args):
    seed_tensor, tensor_list = prepareInputTensorsToRandomTopoTest(seed, args.max_num_tensor, args.max_tensor_dim, args.max_tensor_size, args.debug_tensor, 'cuda' if not args.cpu else 'cpu', torch.float32 if not args.fp16 else torch.float16)
    try:
        if DEBUG_PRINT:
            print('seed tensor: ', seed_tensor)
        o = random_topology_test(seed_tensor, *tensor_list)
        if DEBUG_PRINT:
            for out in o:
                print('val size: ', out.size())
    except Exception as err:
        raise Exception(f'Testing script failure with error message, repro by running:\n\t{reproString(seed, args)}') from err
    try:
        traced_model = torch.jit.trace(random_topology_test, (seed_tensor, *tensor_list))
        if DEBUG_PRINT:
            print('original graph: ', traced_model.graph)
        jit_o = traced_model(seed_tensor, *tensor_list)
        jit_o = traced_model(seed_tensor, *tensor_list)
        if DEBUG_PRINT:
            print('optimized graph: ', traced_model.graph_for(seed_tensor, *tensor_list))
        validate_o = zip(o, jit_o)
        for oo, jit_oo in validate_o:
            if not oo.allclose(jit_oo, equal_nan=True):
                print('eager output: ', oo)
                print('jit output: ', jit_oo)
                print('diff ', jit_oo - oo)
                raise WrongResultException()
    except WrongResultException as err:
        raise Exception(f'cuda fuser gives wrong results, repro by running:\n\t{reproString(seed, args)}') from err
    except Exception as err:
        raise Exception(f'something in cuda fuser went wrong, repro by running:\n\t{reproString(seed, args)}') from err
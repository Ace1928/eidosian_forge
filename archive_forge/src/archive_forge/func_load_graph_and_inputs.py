from contextlib import contextmanager
from typing import Any, List, Tuple, cast
import random
import torch
import time
from torch.utils.benchmark import Timer
def load_graph_and_inputs(ir: str) -> Tuple[Any, List[Any]]:
    graph = torch._C.parse_ir(ir, parse_tensor_constants=True)
    graph.makeMultiOutputIntoTuple()
    inputs = []
    for inp in graph.inputs():
        if isinstance(inp.type(), torch._C.FloatType):
            inputs.append(random.uniform(0.1, 100))
        elif isinstance(inp.type(), torch._C.IntType):
            inputs.append(random.randint(1, 100))
        elif isinstance(inp.type(), torch._C.TensorType):
            tensorType = cast(torch._C.TensorType, inp.type())
            inputs.append(make_tensor_from_type(tensorType))
        elif isinstance(inp.type(), torch._C.BoolType):
            inputs.append(random.randint(0, 1) == 1)
        else:
            raise NotImplementedError(f'A default value is not implemented for type {inp.type()}')
    func = torch._C._create_function_from_graph('forward', graph)
    torch._C._jit_pass_erase_shape_information(func.graph)
    return (func, inputs)
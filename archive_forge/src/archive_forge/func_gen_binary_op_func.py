import builtins
import torch
from torch.distributed._shard.sharding_spec import (
from torch.distributed._shard.sharding_spec._internals import (
def gen_binary_op_func(python_op, inplace=False):
    src_lines = ['def f(lhs, rhs):']
    if 'torch' in python_op:
        src_lines.append(f'  return {python_op}(lhs, rhs)\n')
    elif inplace:
        src_lines.append(f'  lhs {python_op}= rhs\n  return lhs\n')
    else:
        src_lines.append(f'  return lhs {python_op} rhs\n')
    code_str = '\n'.join(src_lines)
    g = {'torch': torch}
    builtins.exec(code_str, g)
    return g['f']
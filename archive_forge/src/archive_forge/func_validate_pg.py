import torch
from torch.utils import _pytree as pytree
from typing import Optional
def validate_pg(e):
    nonlocal cur_pg
    if isinstance(e, ShardedTensor):
        if cur_pg is not None and e._process_group is not cur_pg:
            raise RuntimeError('All distributed tensors should use the same ProcessGroup if used together in an op.')
        cur_pg = e._process_group
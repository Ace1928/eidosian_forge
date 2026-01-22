import collections
import contextlib
import re
import torch
from typing import Callable, Dict, Optional, Tuple, Type, Union
def collate_tensor_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]]=None):
    elem = batch[0]
    out = None
    if elem.is_nested:
        raise RuntimeError('Batches of nested tensors are not currently supported by the default collate_fn; please provide a custom collate_fn to handle them appropriately.')
    if elem.layout in {torch.sparse_coo, torch.sparse_csr, torch.sparse_bsr, torch.sparse_csc, torch.sparse_bsc}:
        raise RuntimeError('Batches of sparse tensors are not currently supported by the default collate_fn; please provide a custom collate_fn to handle them appropriately.')
    if torch.utils.data.get_worker_info() is not None:
        numel = sum((x.numel() for x in batch))
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)
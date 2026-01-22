from __future__ import annotations
import logging
from typing import Optional, Sequence
import torch
from torch import _prims, Tensor
def make_prim(schema: str, impl_aten, return_type=_prims.RETURN_TYPE.NEW, doc: str='', tags: Optional[Sequence[torch.Tag]]=None):

    def meta(*args, **kwargs):
        return _prims.TensorMeta(impl_aten(*args, **kwargs))
    return _prims._make_prim(schema=schema, return_type=return_type, meta=meta, impl_aten=impl_aten, doc=doc, tags=tags)
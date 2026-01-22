from typing import Optional, Sequence
import torch
from xformers.ops._triton import (
from .common import BaseOperator, register_operator
def index_select_cat(sources: Sequence[torch.Tensor], indices: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Indices in ``index`` are assumed to be unique
    In each (index, source) pair, the max index in ``index`` is assumed to be less than the size of dim0 of ``source``

    :Example:

    Given:
    - ``sources[0]`` of shape ``[S0, D0]``
    - ``indices[0]`` of shape ``[I0]``
    - ``sources[1]`` of shape ``[S1, D1]``
    - ``indices[1]`` of shape ``[I1]``
    returns a ``torch.Tensor`` of shape ``[I0 * D0 + I1 * D1]``

    :Equivalent pytorch code:

    .. code-block:: python

        return torch.cat([s[i.long()].flatten() for s, i in zip(sources, indices)], dim=0)
    """
    return _IndexSelectCat.apply(*sources, *indices)
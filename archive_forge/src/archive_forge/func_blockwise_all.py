from __future__ import annotations
from typing import (
from pandas.core.dtypes.common import is_1d_only_ea_dtype
def blockwise_all(left: BlockManager, right: BlockManager, op) -> bool:
    """
    Blockwise `all` reduction.
    """
    for info in _iter_block_pairs(left, right):
        res = op(info.lvals, info.rvals)
        if not res:
            return False
    return True
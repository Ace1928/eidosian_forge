import numpy
import math
import types as pytypes
import collections
import warnings
import numba
from numba.core.extending import _Intrinsic
from numba.core import types, typing, ir, analysis, postproc, rewrites, config
from numba.core.typing.templates import signature
from numba.core.analysis import (compute_live_map, compute_use_defs,
from numba.core.errors import (TypingError, UnsupportedError,
import copy
def merge_adjacent_blocks(blocks):
    cfg = compute_cfg_from_blocks(blocks)
    removed = set()
    for label in list(blocks.keys()):
        if label in removed:
            continue
        block = blocks[label]
        succs = list(cfg.successors(label))
        while True:
            if len(succs) != 1:
                break
            next_label = succs[0][0]
            if next_label in removed:
                break
            preds = list(cfg.predecessors(next_label))
            succs = list(cfg.successors(next_label))
            if len(preds) != 1 or preds[0][0] != label:
                break
            next_block = blocks[next_label]
            block.body.pop()
            block.body += next_block.body
            del blocks[next_label]
            removed.add(next_label)
            label = next_label
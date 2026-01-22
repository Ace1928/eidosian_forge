import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
def one_layer_rnn_data(inp, hidden, params, has_biases, hidden_fn, batch_sizes, reverse=False):
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    step_output = []
    hiddens: List[torch.Tensor] = []
    last_batch_size = batch_sizes[-1] if reverse else batch_sizes[0]
    cur_hidden = hidden.narrow(0, 0, last_batch_size)
    split_inp = torch.split(inp, list(batch_sizes))
    if reverse:
        split_inp = split_inp[::-1]
    for inp in split_inp:
        i = inp.shape[0]
        if last_batch_size == i:
            pass
        elif reverse:
            cur_hidden = update_hidden_for_packed_reverse(cur_hidden, last_batch_size, i, hidden)
        else:
            cur_hidden = update_hidden_for_packed(cur_hidden, last_batch_size, i, hiddens)
        cur_hidden = hidden_fn(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias)
        last_batch_size = i
        step_output.append(cur_hidden)
    if reverse:
        step_output.reverse()
    else:
        hiddens.append(cur_hidden)
        hiddens.reverse()
    out = torch.cat(step_output, 0)
    hidden_out = torch.cat(hiddens, 0) if not reverse else cur_hidden
    return (out, hidden_out)
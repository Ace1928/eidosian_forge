import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta(aten.mkldnn_rnn_layer_backward.default)
def mkldnn_rnn_layer_backward(input, weight0, weight1, weight2, weight3, hx_, cx_tmp, output, hy_, cy_, grad_output_r_opt, grad_hy_r_opt, grad_cy_r_opt, reverse, mode, hidden_size, num_layers, has_biases, train, bidirectional, batch_sizes, batch_first, workspace):
    diff_x = input.new_empty(input.shape)
    diff_hx = hx_.new_empty(hx_.shape)
    diff_cx = cx_tmp.new_empty(cx_tmp.shape)
    diff_w1 = weight0.new_empty(weight0.shape)
    diff_w2 = weight1.new_empty(weight1.shape)
    diff_b = weight2.new_empty(weight2.shape)
    return (diff_x, diff_w1, diff_w2, diff_b, diff_b, diff_hx, diff_cx)
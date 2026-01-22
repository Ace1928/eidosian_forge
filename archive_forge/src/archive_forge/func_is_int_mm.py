import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import (
from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._higher_order_ops.utils import autograd_not_implemented
def is_int_mm(op, output_dtype, args):
    return op == torch.ops.aten.mm.default and output_dtype == torch.int32 and (len(args) == 2) and (args[0].dtype == torch.int8) and (args[1].dtype == torch.int8) and args[0].is_cuda and args[1].is_cuda
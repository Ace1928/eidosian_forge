import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._dispatch.python import suspend_functionalization
from torch._functorch.aot_autograd import AOTConfig, create_joint
from torch._functorch.eager_transforms import (
from torch._higher_order_ops.cond import (
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import (
@map_impl.py_impl(DispatchKey.CompositeExplicitAutograd)
def map_dense(f, num_mapped_args, *args):
    xs = args[:num_mapped_args]
    pos_args = args[num_mapped_args:]
    pytrees = []
    for inp in _unstack_pytree(xs):
        pytrees.append(f(*inp, *pos_args))
    return _stack_pytree(pytrees)
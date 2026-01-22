from contextlib import contextmanager
import torch
import torch._custom_ops
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree
def update_module_call_signatures(path, in_spec, out_spec):
    assert path not in module_call_specs
    module_call_specs[path] = {'in_spec': in_spec, 'out_spec': out_spec}
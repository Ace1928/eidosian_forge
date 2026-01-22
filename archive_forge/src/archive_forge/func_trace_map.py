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
def trace_map(proxy_mode, func_overload, f, num_mapped, *args):
    xs = list(args[:num_mapped])
    pos_args = list(args[num_mapped:])
    leading_dim_size = xs[0].shape[0]
    example_input = _unstack_pytree(xs)[0]
    body_graph = f
    if not isinstance(body_graph, torch.fx.GraphModule):
        body_graph = make_fx(body_graph)(*example_input, *pos_args)
    with disable_proxy_modes_tracing():
        example_outs = body_graph(*example_input, *pos_args)

        def expand_tensor(t):
            if isinstance(t, torch.Tensor):
                return t.expand(leading_dim_size, *t.shape)
            return t
        expanded_outs = pytree.tree_map(expand_tensor, example_outs)
    next_name = None
    i = 0
    while not next_name:
        candidate = f'body_graph_{i}'
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate
    proxy_mode.tracer.root.register_module(next_name, body_graph)
    node_args = (body_graph, num_mapped, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {}, name='map_impl')
    return track_tensor_tree(expanded_outs, out_proxy, constant=None, tracer=proxy_mode.tracer)
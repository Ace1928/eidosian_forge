from contextlib import contextmanager
from dataclasses import dataclass
import torch
import torch._subclasses.functional_tensor
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import _get_current_dispatch_mode
def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    assert isinstance(operands, (list, tuple)), 'Cond operands must be a list or tuple of tensors'
    assert all((isinstance(o, torch.Tensor) for o in operands)), 'Cond operands must be a list of tensors'
    pre_dispatch = getattr(proxy_mode, 'pre_dispatch', False)
    with disable_proxy_modes_tracing():
        true_graph = make_fx(_maybe_run_with_interpreter(true_fn), pre_dispatch=pre_dispatch)(*operands)
        false_graph = make_fx(_maybe_run_with_interpreter(false_fn), pre_dispatch=pre_dispatch)(*operands)
    true_outs = []
    false_outs = []
    for node in true_graph.graph.nodes:
        if node.op == 'output':
            true_outs.extend(node.args)
    for node in false_graph.graph.nodes:
        if node.op == 'output':
            false_outs.extend(node.args)
    flat_true_outs = pytree.arg_tree_leaves(*true_outs)
    flat_false_outs = pytree.arg_tree_leaves(*false_outs)
    if len(flat_true_outs) != len(flat_false_outs):
        raise torch._dynamo.exc.CondOpArgsMismatchError(f'Expected to return same number of outputs but got:\n  {true_fn.__name__} returns {len(flat_true_outs)} item(s)\n  {false_fn.__name__} returns {len(flat_false_outs)} item(s)')
    for i in range(0, len(flat_true_outs)):
        true_out = flat_true_outs[i]
        false_out = flat_false_outs[i]
        if true_out.meta['tensor_meta'] != false_out.meta['tensor_meta']:
            raise torch._dynamo.exc.CondOpArgsMismatchError(f'Expected each tensor to have same metadata but got:\n  {true_fn.__name__} returns {true_out.meta['tensor_meta']}\n  {false_fn.__name__} returns {false_out.meta['tensor_meta']}')
    next_name = None
    i = 0
    while not next_name:
        candidate = f'true_graph_{i}'
        if hasattr(proxy_mode.tracer.root, candidate):
            i += 1
        else:
            next_name = candidate
    true_name = next_name
    false_name = f'false_graph_{i}'
    assert not hasattr(proxy_mode.tracer.root, false_name)
    proxy_mode.tracer.root.register_module(true_name, true_graph)
    proxy_mode.tracer.root.register_module(false_name, false_graph)
    args = (pred, true_graph, false_graph, operands)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', func_overload, proxy_args, {}, name='conditional')
    out = false_fn(*operands)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
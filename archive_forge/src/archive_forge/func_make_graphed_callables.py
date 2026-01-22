import gc
import torch
from torch.utils import _pytree
from ._utils import _dummy_type
from torch._C import (  # noqa: F401
def make_graphed_callables(callables, sample_args, num_warmup_iters=3, allow_unused_input=False):
    """Accept callables (functions or :class:`nn.Module<torch.nn.Module>`\\ s) and returns graphed versions.

    Each graphed callable's forward pass runs its source callable's
    forward CUDA work as a CUDA graph inside a single autograd node.

    The graphed callable's forward pass also appends
    a backward node to the autograd graph. During backward, this node runs the
    callable's backward work as a CUDA graph.

    Therefore, each graphed callable should be a drop-in replacement for its source callable
    in an autograd-enabled training loop.

    See :ref:`Partial-network capture<partial-network-capture>` for detailed use and constraints.

    If you pass a tuple of several callables, their captures will use the same memory pool.
    See :ref:`Graph memory management<graph-memory-management>` for when this is appropriate.

    Arguments:
        callables (torch.nn.Module or Python function, or tuple of these): Callable or callables to graph.
            See :ref:`Graph memory management<graph-memory-management>` for when passing a tuple of callables
            is appropriate.  If you pass a tuple of callables, their order in the tuple must be the same order
            they'll run in the live workload.
        sample_args (tuple of Tensors, or tuple of tuples of Tensors): Samples args for each callable.
            If a single callable was passed, ``sample_args`` must be a single tuple of argument Tensors.
            If a tuple of callables was passed, ``sample_args`` must be tuple of tuples of argument Tensors.
        num_warmup_iters (int): The number of warmup iterations. Currently, ``DataDistributedParallel`` needs
            11 iterations for warm up. Default: ``3``.
        allow_unused_input (bool): If False, specifying inputs that were not used when computing outputs
            (and therefore their grad is always zero) is an error. Defaults to False.

    .. note::
        The ``requires_grad`` state of each Tensor in ``sample_args`` must match the state
        that's expected for the corresponding real input in the training loop.

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        ``sample_args`` for each callable must contain only Tensors. Other types are not allowed.

    .. warning::
        Returned callables do not support higher order differentiation (e.g., double backward).

    .. warning::
        In any :class:`~torch.nn.Module` passed to :func:`~make_graphed_callables`, only parameters
        may be trainable. Buffers must have ``requires_grad=False``.

    .. warning::
        After you pass a :class:`torch.nn.Module` through :func:`~make_graphed_callables`,
        you may not add or remove any of that Module's parameters or buffers.

    .. warning::
        :class:`torch.nn.Module`\\s passed to :func:`~torch.cuda.make_graphed_callables` must not have module hooks
        registered on them at the time they are passed. However, registering hooks on modules *after* passing them
        through :func:`~torch.cuda.make_graphed_callables` is allowed.

    .. warning::
        When running a graphed callable, you must pass its arguments in the same order and format
        they appeared in that callable's ``sample_args``.

    .. warning::
        The automatic mixed precision is supported in :func:`~torch.cuda.make_graphed_callables` only with disabled
        caching. The context manager `torch.cuda.amp.autocast()` must have `cache_enabled=False`.
    """
    if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
        raise RuntimeError('make_graphed_callables does not support the autocast caching. Please set `cache_enabled=False`.')
    just_one_callable = False
    if not isinstance(callables, tuple):
        just_one_callable = True
        callables = (callables,)
        sample_args = (sample_args,)
    flatten_sample_args = []
    for c, args in zip(callables, sample_args):
        if isinstance(c, torch.nn.Module):
            assert len(c._backward_hooks) == 0 and len(c._forward_hooks) == 0 and (len(c._forward_pre_hooks) == 0), 'Modules must not have hooks registered at the time they are passed. However, registering hooks ' + 'on modules after passing them through make_graphed_callables is allowed.'
            assert all((b.requires_grad is False for b in c.buffers())), 'In any :class:`~torch.nn.Module` passed to ' + ':func:`~make_graphed_callables`, only parameters may be trainable. All buffers must have ' + '``requires_grad=False``.'
        flatten_arg = _pytree.arg_tree_leaves(*args)
        flatten_sample_args.append(tuple(flatten_arg))
        assert all((isinstance(arg, torch.Tensor) for arg in flatten_arg)), 'In the beta API, sample_args ' + 'for each callable must contain only Tensors. Other types are not allowed.'
    per_callable_len_user_args = [len(args) for args in flatten_sample_args]
    per_callable_module_params = [tuple(c.parameters()) if isinstance(c, torch.nn.Module) else () for c in callables]
    per_callable_static_input_surfaces = [flatten_sample_args[i] + per_callable_module_params[i] for i in range(len(callables))]
    fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
    mempool = graph_pool_handle()
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream()):
        for func, args, static_input_surface in zip(callables, sample_args, per_callable_static_input_surfaces):
            for _ in range(num_warmup_iters):
                outputs = _pytree.tree_leaves(func(*args))
                grad_inputs = torch.autograd.grad(outputs=tuple((o for o in outputs if o.requires_grad)), inputs=tuple((i for i in static_input_surface if i.requires_grad)), grad_outputs=tuple((torch.empty_like(o) for o in outputs if o.requires_grad)), only_inputs=True, allow_unused=allow_unused_input)
            del outputs, grad_inputs
    torch.cuda.synchronize()
    per_callable_static_outputs = []
    per_callable_output_unflatten_spec = []
    for func, args, fwd_graph in zip(callables, sample_args, fwd_graphs):
        with torch.cuda.graph(fwd_graph, pool=mempool):
            outputs = func(*args)
        flatten_outputs, spec = _pytree.tree_flatten(outputs)
        per_callable_static_outputs.append(tuple(flatten_outputs))
        per_callable_output_unflatten_spec.append(spec)
    per_callable_static_grad_outputs = []
    per_callable_static_grad_inputs = []
    for static_input_surface, static_outputs, bwd_graph, module_params in zip(reversed(per_callable_static_input_surfaces), reversed(per_callable_static_outputs), reversed(bwd_graphs), reversed(per_callable_module_params)):
        static_grad_outputs = tuple((torch.empty_like(o) if o.requires_grad else None for o in static_outputs))
        with torch.cuda.graph(bwd_graph, pool=mempool):
            grad_inputs = torch.autograd.grad(outputs=tuple((o for o in static_outputs if o.requires_grad)), inputs=tuple((i for i in static_input_surface if i.requires_grad)), grad_outputs=tuple((o for o in static_grad_outputs if o is not None)), only_inputs=True, allow_unused=allow_unused_input)
        static_grad_inputs = []
        grad_idx = 0
        for arg in static_input_surface:
            if arg.requires_grad:
                static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)
        static_grad_inputs = tuple(static_grad_inputs)
        per_callable_static_grad_outputs.append(static_grad_outputs)
        per_callable_static_grad_inputs.append(static_grad_inputs)
    per_callable_static_grad_outputs = list(reversed(per_callable_static_grad_outputs))
    per_callable_static_grad_inputs = list(reversed(per_callable_static_grad_inputs))

    def make_graphed_autograd_function(fwd_graph, bwd_graph, module_params, len_user_args, output_unflatten_spec, static_input_surface, static_outputs, static_grad_outputs, static_grad_inputs):

        class Graphed(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *inputs):
                for i in range(len_user_args):
                    if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                        static_input_surface[i].copy_(inputs[i])
                fwd_graph.replay()
                assert isinstance(static_outputs, tuple)
                return tuple((o.detach() for o in static_outputs))

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, *grads):
                assert len(grads) == len(static_grad_outputs)
                for g, grad in zip(static_grad_outputs, grads):
                    if g is not None:
                        if g.data_ptr() != grad.data_ptr():
                            g.copy_(grad)
                bwd_graph.replay()
                assert isinstance(static_grad_inputs, tuple)
                return tuple((b.detach() if b is not None else b for b in static_grad_inputs))

        def functionalized(*user_args):
            flatten_user_args = _pytree.arg_tree_leaves(*user_args)
            out = Graphed.apply(*tuple(flatten_user_args) + module_params)
            return _pytree.tree_unflatten(out, output_unflatten_spec)
        return functionalized
    ret = []
    for i, func in enumerate(callables):
        graphed = make_graphed_autograd_function(fwd_graphs[i], bwd_graphs[i], per_callable_module_params[i], per_callable_len_user_args[i], per_callable_output_unflatten_spec[i], per_callable_static_input_surfaces[i], per_callable_static_outputs[i], per_callable_static_grad_outputs[i], per_callable_static_grad_inputs[i])
        if isinstance(func, torch.nn.Module):

            def make_graphed_forward(func, graph_training_state, graphed, orig_fwd):

                def new_fwd(*user_args):
                    if func.training == graph_training_state:
                        return graphed(*user_args)
                    else:
                        return orig_fwd(*user_args)
                return new_fwd
            func.forward = make_graphed_forward(func, func.training, graphed, func.forward)
            ret.append(func)
        else:
            ret.append(graphed)
    if just_one_callable:
        return ret[0]
    return tuple(ret)
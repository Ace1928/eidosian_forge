import inspect
import math
import re
import warnings
from collections import OrderedDict
from copy import deepcopy
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torchvision
from torch import fx, nn
from torch.fx.graph_module import _copy_attr
def create_feature_extractor(model: nn.Module, return_nodes: Optional[Union[List[str], Dict[str, str]]]=None, train_return_nodes: Optional[Union[List[str], Dict[str, str]]]=None, eval_return_nodes: Optional[Union[List[str], Dict[str, str]]]=None, tracer_kwargs: Optional[Dict[str, Any]]=None, suppress_diff_warning: bool=False) -> fx.GraphModule:
    """
    Creates a new graph module that returns intermediate nodes from a given
    model as dictionary with user specified keys as strings, and the requested
    outputs as values. This is achieved by re-writing the computation graph of
    the model via FX to return the desired nodes as outputs. All unused nodes
    are removed, together with their corresponding parameters.

    Desired output nodes must be specified as a ``.`` separated
    path walking the module hierarchy from top level module down to leaf
    operation or leaf module. For more details on the node naming conventions
    used here, please see the :ref:`relevant subheading <about-node-names>`
    in the `documentation <https://pytorch.org/vision/stable/feature_extraction.html>`_.

    Not all models will be FX traceable, although with some massaging they can
    be made to cooperate. Here's a (not exhaustive) list of tips:

        - If you don't need to trace through a particular, problematic
          sub-module, turn it into a "leaf module" by passing a list of
          ``leaf_modules`` as one of the ``tracer_kwargs`` (see example below).
          It will not be traced through, but rather, the resulting graph will
          hold a reference to that module's forward method.
        - Likewise, you may turn functions into leaf functions by passing a
          list of ``autowrap_functions`` as one of the ``tracer_kwargs`` (see
          example below).
        - Some inbuilt Python functions can be problematic. For instance,
          ``int`` will raise an error during tracing. You may wrap them in your
          own function and then pass that in ``autowrap_functions`` as one of
          the ``tracer_kwargs``.

    For further information on FX see the
    `torch.fx documentation <https://pytorch.org/docs/stable/fx.html>`_.

    Args:
        model (nn.Module): model on which we will extract the features
        return_nodes (list or dict, optional): either a ``List`` or a ``Dict``
            containing the names (or partial names - see note above)
            of the nodes for which the activations will be returned. If it is
            a ``Dict``, the keys are the node names, and the values
            are the user-specified keys for the graph module's returned
            dictionary. If it is a ``List``, it is treated as a ``Dict`` mapping
            node specification strings directly to output names. In the case
            that ``train_return_nodes`` and ``eval_return_nodes`` are specified,
            this should not be specified.
        train_return_nodes (list or dict, optional): similar to
            ``return_nodes``. This can be used if the return nodes
            for train mode are different than those from eval mode.
            If this is specified, ``eval_return_nodes`` must also be specified,
            and ``return_nodes`` should not be specified.
        eval_return_nodes (list or dict, optional): similar to
            ``return_nodes``. This can be used if the return nodes
            for train mode are different than those from eval mode.
            If this is specified, ``train_return_nodes`` must also be specified,
            and `return_nodes` should not be specified.
        tracer_kwargs (dict, optional): a dictionary of keyword arguments for
            ``NodePathTracer`` (which passes them onto it's parent class
            `torch.fx.Tracer <https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer>`_).
            By default, it will be set to wrap and make leaf nodes all torchvision ops:
            {"autowrap_modules": (math, torchvision.ops,),"leaf_modules": _get_leaf_modules_for_ops(),}
            WARNING: In case the user provides tracer_kwargs, above default arguments will be appended to the user
            provided dictionary.
        suppress_diff_warning (bool, optional): whether to suppress a warning
            when there are discrepancies between the train and eval version of
            the graph. Defaults to False.

    Examples::

        >>> # Feature extraction with resnet
        >>> model = torchvision.models.resnet18()
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> model = create_feature_extractor(
        >>>     model, {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = model(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]

        >>> # Specifying leaf modules and leaf functions
        >>> def leaf_function(x):
        >>>     # This would raise a TypeError if traced through
        >>>     return int(x)
        >>>
        >>> class LeafModule(torch.nn.Module):
        >>>     def forward(self, x):
        >>>         # This would raise a TypeError if traced through
        >>>         int(x.shape[0])
        >>>         return torch.nn.functional.relu(x + 4)
        >>>
        >>> class MyModule(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.conv = torch.nn.Conv2d(3, 1, 3)
        >>>         self.leaf_module = LeafModule()
        >>>
        >>>     def forward(self, x):
        >>>         leaf_function(x.shape[0])
        >>>         x = self.conv(x)
        >>>         return self.leaf_module(x)
        >>>
        >>> model = create_feature_extractor(
        >>>     MyModule(), return_nodes=['leaf_module'],
        >>>     tracer_kwargs={'leaf_modules': [LeafModule],
        >>>                    'autowrap_functions': [leaf_function]})

    """
    tracer_kwargs = _set_default_tracer_kwargs(tracer_kwargs)
    is_training = model.training
    if all((arg is None for arg in [return_nodes, train_return_nodes, eval_return_nodes])):
        raise ValueError('Either `return_nodes` or `train_return_nodes` and `eval_return_nodes` together, should be specified')
    if (train_return_nodes is None) ^ (eval_return_nodes is None):
        raise ValueError('If any of `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified')
    if not (return_nodes is None) ^ (train_return_nodes is None):
        raise ValueError('If `train_return_nodes` and `eval_return_nodes` are specified, then both should be specified')

    def to_strdict(n) -> Dict[str, str]:
        if isinstance(n, list):
            return {str(i): str(i) for i in n}
        return {str(k): str(v) for k, v in n.items()}
    if train_return_nodes is None:
        return_nodes = to_strdict(return_nodes)
        train_return_nodes = deepcopy(return_nodes)
        eval_return_nodes = deepcopy(return_nodes)
    else:
        train_return_nodes = to_strdict(train_return_nodes)
        eval_return_nodes = to_strdict(eval_return_nodes)
    tracers = {}
    graphs = {}
    mode_return_nodes: Dict[str, Dict[str, str]] = {'train': train_return_nodes, 'eval': eval_return_nodes}
    for mode in ['train', 'eval']:
        if mode == 'train':
            model.train()
        elif mode == 'eval':
            model.eval()
        tracer = NodePathTracer(**tracer_kwargs)
        graph = tracer.trace(model)
        name = model.__class__.__name__ if isinstance(model, nn.Module) else model.__name__
        graph_module = fx.GraphModule(tracer.root, graph, name)
        available_nodes = list(tracer.node_to_qualname.values())
        if len(set(available_nodes)) != len(available_nodes):
            raise ValueError('There are duplicate nodes! Please raise an issue https://github.com/pytorch/vision/issues')
        for query in mode_return_nodes[mode].keys():
            if not any([re.match(f'^{query}(\\.|$)', n) is not None for n in available_nodes]):
                raise ValueError(f"node: '{query}' is not present in model. Hint: use `get_graph_node_names` to make sure the `return_nodes` you specified are present. It may even be that you need to specify `train_return_nodes` and `eval_return_nodes` separately.")
        orig_output_nodes = []
        for n in reversed(graph_module.graph.nodes):
            if n.op == 'output':
                orig_output_nodes.append(n)
        if not orig_output_nodes:
            raise ValueError('No output nodes found in graph_module.graph.nodes')
        for n in orig_output_nodes:
            graph_module.graph.erase_node(n)
        nodes = [n for n in graph_module.graph.nodes]
        output_nodes = OrderedDict()
        for n in reversed(nodes):
            module_qualname = tracer.node_to_qualname.get(n)
            if module_qualname is None:
                continue
            for query in mode_return_nodes[mode]:
                depth = query.count('.')
                if '.'.join(module_qualname.split('.')[:depth + 1]) == query:
                    output_nodes[mode_return_nodes[mode][query]] = n
                    mode_return_nodes[mode].pop(query)
                    break
        output_nodes = OrderedDict(reversed(list(output_nodes.items())))
        with graph_module.graph.inserting_after(nodes[-1]):
            graph_module.graph.output(output_nodes)
        graph_module.graph.eliminate_dead_code()
        graph_module.recompile()
        tracers[mode] = tracer
        graphs[mode] = graph
    if not suppress_diff_warning:
        _warn_graph_differences(tracers['train'], tracers['eval'])
    graph_module = DualGraphModule(model, graphs['train'], graphs['eval'], class_name=name)
    model.train(is_training)
    graph_module.train(is_training)
    return graph_module
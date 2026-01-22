import torch.fx as fx
from torch.fx.node import Argument, Target
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp
import copy
from collections import defaultdict
import torch.utils.mkldnn as th_mkldnn
import operator
import time
import logging
from enum import Enum
def use_mkl_heuristic(graph: MklSubgraph) -> bool:
    nonlocal fx_model, old_modules
    input_nodes = graph.start_nodes
    if fx_model is None:
        fx_model = graph.fx_graph.owning_module
        old_modules = graph.fx_graph.old_modules
        ShapeProp(fx_model).propagate(example_inputs)
    sample_inputs = [torch.randn(node.shape) for node in input_nodes]
    output_args = cast(List[fx.Node], [node.args[0] for node in graph.end_nodes])
    submodule = extract_subgraph(fx_model, graph.nodes, input_nodes, output_args)

    def benchmark(f):
        for _ in range(warmup):
            f()
        begin = time.time()
        for _ in range(iters):
            out = f()
        return time.time() - begin
    mkl_time = benchmark(lambda: [i.to_dense() for i in submodule(*[i.to_mkldnn() for i in sample_inputs])])
    reset_modules(submodule.graph.nodes, dict(submodule.named_modules()), old_modules)
    no_mkl_time = benchmark(lambda: submodule(*sample_inputs))
    return mkl_time < no_mkl_time
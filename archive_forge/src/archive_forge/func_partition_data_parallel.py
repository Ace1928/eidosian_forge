import operator
from contextlib import contextmanager
from enum import Enum
from typing import Any, cast, Dict, List, Optional, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
import torch.fx as fx
import torch.library
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.distributed._spmd.batch_dim_utils import BatchDimAnalyzer
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec, Placement
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def partition_data_parallel(graph: GraphModule, model: nn.Module, optimizer: Optional[torch.optim.Optimizer], params_buffers: Dict[str, torch.Tensor], named_states: Dict[str, Any], args: Tuple[Any, ...], kwargs: Dict[str, Any], mesh: DeviceMesh, parallel_style: DataParallelStyle, input_batch_dim: int) -> GraphModule:
    """Partition the graph to into a data parallel graph.

    This function also shards/replicates the model parameters and optimizer states to DTensors.
    """
    num_params_buffers = len(params_buffers)
    flattened_states = pytree.tree_leaves(named_states)
    num_states = len(flattened_states)
    changed = graph.graph.eliminate_dead_code()
    if changed:
        graph.recompile()
    strategy_map = build_data_parallel_strategies(graph, num_params_buffers, num_states, mesh=mesh, batch_dim=input_batch_dim)
    mark_data_parallel_shardings(graph, num_parameters=num_params_buffers, num_states=num_states, dp_strategy_map=strategy_map, parallel_mode=parallel_style)
    partitioned_graph = partitioner(graph)
    for node in partitioned_graph.graph.nodes:
        if node in strategy_map:
            node_strategy = strategy_map[node]
            if isinstance(node_strategy, DataParallelStrategy):
                node.meta['node_type'] = node_strategy.node_type
            elif isinstance(node_strategy, TupleStrategy):
                node.meta['node_type'] = NodeType.NON_TENSOR
            else:
                raise RuntimeError(f'Unknown node strategy {node_strategy}')
        else:
            input_node = node.all_input_nodes[0]
            node.meta['node_type'] = input_node.meta['node_type']
    accessor = NamedMemberAccessor(model)
    for param_key, param in params_buffers.items():
        placement: Placement = Replicate()
        if parallel_style == DataParallelStyle.FULLY_SHARD:
            placement = Shard(0)
        elif parallel_style != DataParallelStyle.REPLICATE:
            raise RuntimeError(f'parallel style {parallel_style} not supported yet')
        dtensor_param = distribute_tensor(param, mesh, [placement])
        params_buffers[param_key] = dtensor_param.to_local()
        accessor.set_tensor(param_key, dtensor_param)
        if optimizer is not None and param in optimizer.state:
            param_states = named_states[param_key]
            param_dtensor_states = {}
            for state_key, state_val in param_states.items():
                if isinstance(state_val, torch.Tensor) and state_val.ndim > 0:
                    dtensor_state = distribute_tensor(state_val, mesh, [placement])
                    param_dtensor_states[state_key] = dtensor_state
                    param_states[state_key] = dtensor_state.to_local()
                else:
                    param_dtensor_states[state_key] = state_val
            optimizer.state.pop(param)
            optimizer.state[dtensor_param] = param_dtensor_states
    return partitioned_graph
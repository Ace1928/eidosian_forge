import logging
from typing import Dict, List, Set
import torch
import torch.fx
from torch.fx.node import Node
def shard_model(model: torch.nn.Module, shard_count: int=3) -> List[torch.fx.GraphModule]:
    """Utility used to shard a model using torch.fx.

    This function traces the model twice in an attempt to identify the
    right cutpoints and then shard the model. In the first pass we calculate
    the number of parameters as we are tracing the graph and mark nodes at
    which we might want to create a new module. In the second pass we
    modify the graph by inserting placeholders and output nodes to essentially
    shard the graph.

    We don't support skip connections between shards. This means that all
    input and output is self contained within a given shard. A node from
    shard 1 cannot be an input to a node from shard 3. We expect all inputs
    to a given shard to be coming from the last node in the previous shard.
    This means that we may not be able to shard models by the specified
    `shard_count` mentioned by the user.

    Args:
        model (nn.Module): Model to be sharded as specified by the device count.

        shard_count (int): Number of shards that we want to split the model into.

    """
    module_list: List[torch.fx.GraphModule] = []
    num_graphs = 0
    new_graph = torch.fx.Graph()
    env: Dict[str, Node] = {}
    new_input_node = None
    traced_graph_module = _trace(model)
    node_name_to_shard_id = _split_nodes(traced_graph_module, shard_count=shard_count)
    prev_shard_id = 1000
    prev_node = None
    for node in traced_graph_module.graph.nodes:
        if node.name in node_name_to_shard_id and prev_shard_id < node_name_to_shard_id[node.name]:
            assert prev_node, 'prev_node cannot be None'
            with new_graph.inserting_after(prev_node):
                new_graph.output(env[prev_node.name])
            num_graphs += 1
            module_list.append(torch.fx.GraphModule(model, new_graph))
            new_graph = torch.fx.Graph()
            node_name = 'placeholder' + str(num_graphs)
            pl_node = new_graph.create_node('placeholder', node_name)
            env[node_name] = pl_node
            new_input_node = pl_node
        if new_input_node is not None:
            node.args = (new_input_node,)
            new_input_node = None
        if node.op in ['placeholder', 'get_attr', 'call_function', 'call_method', 'call_module']:
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        elif node.op == 'output':
            assert prev_node, 'prev_node cannot be None'
            with new_graph.inserting_after(prev_node):
                new_graph.output(env[prev_node.name])
            module_list.append(torch.fx.GraphModule(model, new_graph))
            break
        prev_node = new_node
        prev_shard_id = node_name_to_shard_id[node.name]
    return module_list
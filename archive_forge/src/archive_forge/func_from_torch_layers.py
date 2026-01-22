import itertools
from functools import reduce
from operator import mul
from typing import List
import wandb
from wandb import util
from wandb.data_types import Node
@classmethod
def from_torch_layers(cls, module_graph, variable):
    """Recover something like neural net layers from PyTorch Module's and the
        compute graph from a Variable.

        Example output for a multi-layer RNN. We confusingly assign shared embedding values
        to the encoder, but ordered next to the decoder.

        rnns.0.linear.module.weight_raw rnns.0
        rnns.0.linear.module.bias rnns.0
        rnns.1.linear.module.weight_raw rnns.1
        rnns.1.linear.module.bias rnns.1
        rnns.2.linear.module.weight_raw rnns.2
        rnns.2.linear.module.bias rnns.2
        rnns.3.linear.module.weight_raw rnns.3
        rnns.3.linear.module.bias rnns.3
        decoder.weight encoder
        decoder.bias decoder
        """
    torch = util.get_module('torch', 'Could not import torch')
    module_nodes_by_hash = {id(n): n for n in module_graph.nodes}
    module_parameter_nodes = [n for n in module_graph.nodes if isinstance(n.obj, torch.nn.Parameter)]
    names_by_pid = {id(n.obj): n.name for n in module_parameter_nodes}
    reachable_param_nodes = module_graph[0].reachable_descendents()
    reachable_params = {}
    module_reachable_params = {}
    names = {}
    for pid, reachable_nodes in reachable_param_nodes.items():
        node = module_nodes_by_hash[pid]
        if not isinstance(node.obj, torch.nn.Module):
            continue
        module = node.obj
        reachable_params = {}
        module_reachable_params[id(module)] = reachable_params
        names[node.name] = set()
        for reachable_hash in reachable_nodes:
            reachable = module_nodes_by_hash[reachable_hash]
            if isinstance(reachable.obj, torch.nn.Parameter):
                param = reachable.obj
                reachable_params[id(param)] = param
                names[node.name].add(names_by_pid[id(param)])
    node_depths = {id(n): d for n, d in module_graph[0].descendent_bfs()}
    parameter_module_names = {}
    parameter_modules = {}
    for param_node in (n for n in module_graph.nodes if isinstance(n.obj, torch.nn.Parameter)):
        pid = id(param_node.obj)
        best_node = None
        best_depth = None
        best_reachable_params = None
        for node in module_graph.nodes:
            if not isinstance(node.obj, torch.nn.Module):
                continue
            module = node.obj
            reachable_params = module_reachable_params[id(module)]
            if pid in reachable_params:
                depth = node_depths[id(node)]
                if best_node is None or (len(reachable_params), depth) <= (len(best_reachable_params), best_depth):
                    best_node = node
                    best_depth = depth
                    best_reachable_params = reachable_params
        parameter_modules[pid] = best_node
        parameter_module_names[param_node.name] = best_node.name
    reduced_module_graph = cls()
    rmg_ids = itertools.count()
    rmg_root = Node(id=next(rmg_ids), node=module_graph[0])
    reduced_module_graph.add_node(rmg_root)
    reduced_module_graph.root = rmg_root
    rmg_nodes_by_pid = {}
    module_nodes_by_pid = {id(n.obj): n for n in module_graph.nodes}
    compute_graph, compute_node_vars = cls.from_torch_compute_graph(variable)
    for node, _ in reversed(list(compute_graph[0].ancestor_bfs())):
        param = compute_node_vars.get(node.id)
        pid = id(param)
        if not isinstance(param, torch.nn.Parameter):
            continue
        if pid not in module_nodes_by_pid:
            continue
        mid = id(parameter_modules[pid].obj)
        if mid in rmg_nodes_by_pid:
            rmg_module = rmg_nodes_by_pid[mid]
        else:
            rmg_module = rmg_nodes_by_pid[mid] = Node(id=next(rmg_ids), node=module_nodes_by_pid[mid])
            reduced_module_graph.add_node(rmg_module)
            reduced_module_graph.add_edge(rmg_root, rmg_module)
        rmg_param = Node(id=next(rmg_ids), node=module_nodes_by_pid[pid])
        rmg_nodes_by_pid[pid] = rmg_param
        reduced_module_graph.add_node(rmg_param)
        reduced_module_graph.add_edge(rmg_module, rmg_param)
    return reduced_module_graph
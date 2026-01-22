import torch
import torch.fx as fx
from torch.utils._pytree import tree_flatten
from torch.utils import _pytree as pytree
def fx_graph_cse(fx_g: torch.fx.graph.Graph):
    new_graph = fx.Graph()
    env = {}
    hash_env = {}
    token_map = {}
    for n in fx_g.nodes:
        if n.op == 'placeholder' or n.op == 'output' or n.op == 'get_attr' or (get_aten_target(n) in rand_ops):
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
        else:

            def substitute(arg_list):
                arg_list, spec = tree_flatten(arg_list)
                for i in range(len(arg_list)):
                    v = arg_list[i]
                    if isinstance(v, torch.fx.node.Node) and v in env:
                        arg_list[i] = env[v]
                    if isinstance(v, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                        arg_list[i] = v.node
                return (tuple(arg_list), spec)
            args, args_spec = substitute(n.args)
            kwargs, kwargs_spec = substitute(n.kwargs)
            token = {'target': n.target, 'args': args, 'args_spec': args_spec, 'kwargs': kwargs, 'kwargs_spec': kwargs_spec}
            hash_arg = hash((args, kwargs))
            hash_val = (n.target, hash_arg)
            hash_val_in_hash_env = hash_val in hash_env
            if hash_val_in_hash_env and token_map[hash_val] == token:
                env[n] = hash_env[hash_val]
                continue
            new_node = new_graph.node_copy(n, lambda x: env[x])
            env[n] = new_node
            if not hash_val_in_hash_env:
                hash_env[hash_val] = new_node
                token_map[hash_val] = token
    return new_graph
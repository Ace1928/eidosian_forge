import copy
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Tuple, cast
import torch
from fairscale.nn.misc import FlattenParamsWrapper
def flatten_optim_state_dict(sd: Dict) -> Dict:
    """Shard a full optimizer state dict (called by FSDP.get_shard_from_optim_state_dict)"""
    param_id_map = sd['param_id_map']
    local_ids = set(param_id_map.values())
    if None in local_ids:
        local_ids.remove(None)
    if sd['state']:
        new_state: Dict = {local_id: {} for local_id in local_ids}
        singleton_state: Dict = copy.deepcopy(new_state)
    else:
        new_state = {}
    non_tensor_state = {}
    for global_id, buffers in sd['state'].items():
        local_id = param_id_map[global_id]
        for buffer_name, p in buffers.items():
            if is_singleton_tensor(p):
                singleton_state[local_id][buffer_name] = p
            elif torch.is_tensor(p):
                if buffer_name not in new_state[local_id]:
                    new_state[local_id][buffer_name] = []
                new_state[local_id][buffer_name].append(p.reshape(-1))
            elif isinstance(p, list):
                singleton_state[local_id][buffer_name] = p
            else:
                non_tensor_state[buffer_name] = p
    for local_id, state in new_state.items():
        for buffer_name, tensors in state.items():
            new_state[local_id][buffer_name] = torch.cat(tensors)
        new_state[local_id].update(non_tensor_state)
        new_state[local_id].update(singleton_state[local_id])
    new_sd_pg = copy.deepcopy(sd['param_groups'])
    for pg_id, _ in enumerate(sd['param_groups']):
        num_local_params = sum((1 for _ in groupby(param_id_map.values())))
        new_sd_pg[pg_id]['params'] = list(range(num_local_params))
    sd['state'] = new_state
    sd['param_groups'] = new_sd_pg
    del sd['uncollected_local_ids']
    del sd['param_id_map']
    return sd
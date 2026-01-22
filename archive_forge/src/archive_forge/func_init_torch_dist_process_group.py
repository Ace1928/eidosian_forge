from abc import ABC
from collections import defaultdict
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from typing import Callable, List, T
import ray
from ray.actor import ActorHandle
from ray.train._internal.utils import get_address_and_port
from ray.train.constants import DEFAULT_NCCL_SOCKET_IFNAME
from ray.air._internal.torch_utils import get_device
def init_torch_dist_process_group(workers: List[ActorHandle], backend: str='gloo', init_method: str='env') -> List[int]:
    """Initialize a torch distributed process group.

    Note: this util assumes that the order of the workers passed in
    are their global ranks.

    Args:
        workers: A list of TorchDistributedWorker actors.
        backend: The torch distributed backend to use,
            possible choices are "gloo" or "nccl".
        init_method: The initialization method to use,
            possible choices are "env" or "tcp".

    Returns:
        Local ranks on their respective nodes for the list of workers.
    """
    if not dist.is_available():
        raise RuntimeError('Distributed torch is not available.')
    node_and_gpu_ids = ray.get([w.execute.remote(_get_node_and_gpu_ids) for w in workers])
    node_to_workers = defaultdict(list)
    node_to_gpu_ids = defaultdict(set)
    for i, (node_id, gpu_ids) in enumerate(node_and_gpu_ids):
        node_to_workers[node_id].append(i)
        if not isinstance(gpu_ids, list):
            gpu_ids = [gpu_ids]
        for gpu_id in gpu_ids:
            node_to_gpu_ids[node_id].add(gpu_id)
    master_addr, master_port = ray.get(workers[0].execute.remote(get_address_and_port))
    setup_futures = []
    world_size = len(workers)
    local_ranks = []
    for rank, worker in enumerate(workers):
        node_id = node_and_gpu_ids[rank][0]
        local_rank = node_to_workers[node_id].index(rank)
        local_world_size = len(node_to_workers[node_id])
        setup_futures.append(worker.execute.remote(_init_torch_distributed, init_method=init_method, backend=backend, rank=rank, world_size=world_size, local_rank=local_rank, local_world_size=local_world_size, master_addr=master_addr, master_port=master_port, gpu_ids=list(node_to_gpu_ids[node_id])))
        local_ranks.append(local_rank)
    ray.get(setup_futures)
    return local_ranks
import pickle
from typing import Optional, List, Tuple, TYPE_CHECKING
from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.utils import is_hip, set_cuda_visible_devices, get_ip
def initialize_cluster(parallel_config: ParallelConfig, engine_use_ray: bool=False, ray_address: Optional[str]=None) -> Optional['PlacementGroup']:
    """Initialize the distributed cluster probably with Ray.

    Args:
        parallel_config: The configurations for parallel execution.
        engine_use_ray: Whether to use Ray for async engine.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.

    Returns:
        An optional `PlacementGroup`. It includes the specification
        of the resources for each distributed worker. None if Ray is
        not used.
    """
    if parallel_config.worker_use_ray or engine_use_ray:
        if ray is None:
            raise ImportError('Ray is not installed. Please install Ray to use distributed serving.')
        if is_hip():
            ray.init(address=ray_address, ignore_reinit_error=True, num_gpus=parallel_config.world_size)
        else:
            ray.init(address=ray_address, ignore_reinit_error=True)
    if not parallel_config.worker_use_ray:
        assert parallel_config.world_size == 1, 'Ray is required if parallel_config.world_size > 1.'
        return None
    current_placement_group = ray.util.get_current_placement_group()
    if current_placement_group:
        bundles = current_placement_group.bundle_specs
        gpu_bundles = 0
        for bundle in bundles:
            bundle_gpus = bundle.get('GPU', 0)
            if bundle_gpus > 1:
                raise ValueError('Placement group bundle cannot have more than 1 GPU.')
            if bundle_gpus:
                gpu_bundles += 1
        if parallel_config.world_size > gpu_bundles:
            raise ValueError('The number of required GPUs exceeds the total number of available GPUs in the placement group.')
    else:
        num_gpus_in_cluster = ray.cluster_resources().get('GPU', 0)
        if parallel_config.world_size > num_gpus_in_cluster:
            raise ValueError('The number of required GPUs exceeds the total number of available GPUs in the cluster.')
        placement_group_specs = [{'GPU': 1}] * parallel_config.world_size
        current_placement_group = ray.util.placement_group(placement_group_specs)
        ray.get(current_placement_group.ready(), timeout=1800)
    return current_placement_group
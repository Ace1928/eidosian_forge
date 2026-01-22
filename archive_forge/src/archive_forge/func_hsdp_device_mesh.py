from torch.distributed._tensor.device_mesh import init_device_mesh
import os 
def hsdp_device_mesh(replica_group_size, sharding_group_size, device=None):
    """
     Initializes a device mesh for use with Hybrid Sharding strategy in FSDP (HSDP) training.

    This function requires explicit sizes for replica and sharding groups to accommodate models
    whose GPU fit is unknown, providing flexibility in distributed training setups.
    
    Args:
        replica_group_size (int): The size of each replica group. Must be provided to ensure
            the model fits within the available resources.
        sharding_group_size (int): The size of each sharding group that the model can fit. Must be provided to 
            ensure the correct distribution of model parameters.
        device (str, optional): The device to use (e.g., "cuda:0"). If None, defaults to "cuda"
            with the local rank as the device index.

    Returns:
        A device mesh object compatible with FSDP.

    Raises:
        ValueError: If replica_group_size or sharding_group_size are not provided, or if the
            world size is not evenly divisible by the sharding group size.
        RuntimeError: If a valid device mesh cannot be created.

    Usage:
        If your model fits on 4 GPUS, and you have 3 nodes of 8 GPUs, then:
        Sharding_Group_Size = 4
        Replica_Groups_Size = (24 total gpus, 4 per sharding group) = 6 Replica Groups
        >>> device_mesh = initialize_device_mesh(replica_group_size, sharding_group_size)
        >>> sharded_model = FSDP(model, device_mesh=device_mesh, ...)
    """
    if replica_group_size is None or sharding_group_size is None:
        raise ValueError('Both replica_group_size and sharding_group_size must be provided.')
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    device = device or f'cuda'
    if world_size % sharding_group_size != 0:
        raise ValueError(f'World size {world_size} is not evenly divisible by sharding group size {sharding_group_size}.')
    if world_size // sharding_group_size % replica_group_size != 0:
        raise ValueError(f'The calculated number of replica groups is not evenly divisible by replica_group_size {replica_group_size}.')
    device_mesh = init_device_mesh(device, (replica_group_size, sharding_group_size))
    if device_mesh is None:
        raise RuntimeError('Failed to create a valid device mesh.')
    return device_mesh
from torch.utils.data import DataLoader
from ..utils import is_torch_tpu_available
def tpu_spmd_dataloader(dataloader: DataLoader):
    if is_torch_tpu_available():
        import torch_xla.distributed.parallel_loader as pl
        assert isinstance(dataloader, pl.MpDeviceLoader), 'The dataloader must be a `torch_xla.distributed.parallel_loader.MpDeviceLoader`.'
        import torch_xla.distributed.spmd as xs
        sharding_spec = xs.ShardingSpec(xs.get_global_mesh(), ('fsdp', None))
        dataloader._parallel_loader_kwargs['input_sharding'] = sharding_spec
        return dataloader
    else:
        return dataloader
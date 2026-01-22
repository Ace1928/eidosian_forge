from typing import Any, Callable
import torch
import torch.distributed as dist
def fp16_compress_hook(process_group: dist.ProcessGroup, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    compressed_tensor = bucket.buffer().to(torch.float16).div_(world_size)
    fut = dist.all_reduce(compressed_tensor, group=group_to_use, async_op=True).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor
    return fut.then(decompress)
import torch
import torch.distributed as dist
from torch import nn
def quantization_pertensor_hook(process_group: dist.ProcessGroup, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Applies the ``torch.quantize_per_tensor`` logic to DDP using ``allgather``
    protocol. Workers first allgather the scale and zero point of their own
    ``GradBucket`` prior to the quantization. After all workers have that information,
    the first ``then`` callback called ``quantize_and_allgather`` quantizes worker's
    own gradient tensor, and uses ``allgather`` to communicate these across all workers.
    The final ``then`` callback called ``dequantize_and_aggregate``, dequantizes and
    aggregates each quantized gradient tensor locally and returns the mean.

    .. warning ::
        This is experimental, and uses ``allgather`` protocol which is considerably slower than
        ``allreduce`` protocol. It works only with flattened grads.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, quantization_pertensor_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    rank = process_group.rank() if process_group is not None else dist.get_rank()
    world_size = group_to_use.size()
    tensor = bucket.buffer()
    myObserver = torch.ao.quantization.MinMaxObserver().cuda(tensor.device)
    myObserver(tensor)
    s, z = myObserver.calculate_qparams()
    s_and_z = torch.FloatTensor([s, z]).cuda(tensor.device)
    all_ranks_s_and_z = _get_allgather_out_list(s_and_z, world_size)
    fut = dist.all_gather(all_ranks_s_and_z, s_and_z, group=group_to_use, async_op=True).get_future()

    def quantize_and_allgather(fut):
        all_ranks_s_and_z = fut.wait()[0]
        quantized_tensor = _quantize_per_tensor_cuda(tensor, all_ranks_s_and_z[rank][0], all_ranks_s_and_z[rank][1])
        fut = dist.all_gather(_get_allgather_out_list(quantized_tensor, world_size), quantized_tensor, group=group_to_use, async_op=True).get_future()
        return fut.wait()

    def dequantize_and_aggregate(fut):
        all_ranks_quantized_tensor = fut.wait()[0]
        aggregated_dequantized_tensor = torch.zeros_like(all_ranks_quantized_tensor[0], device=tensor.device, dtype=torch.float32)
        for r, quantized_tensor in enumerate(all_ranks_quantized_tensor):
            aggregated_dequantized_tensor += _dequantize_per_tensor_cuda(quantized_tensor, all_ranks_s_and_z[r][0], all_ranks_s_and_z[r][1])
        return aggregated_dequantized_tensor / world_size
    return fut.then(quantize_and_allgather).then(dequantize_and_aggregate)
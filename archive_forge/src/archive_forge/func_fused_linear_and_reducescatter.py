import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def fused_linear_and_reducescatter(gathered_input: torch.Tensor, weight: Union[torch.Tensor, List[torch.Tensor]], *, group: dist.ProcessGroup, out: Optional[Union[torch.Tensor, List[torch.Tensor]]]=None, num_stripes: int=1, timeout_s: int=60 * 60, scale_gathered_input: Optional[torch.Tensor]=None, scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]]=None, out_dtype: Optional[torch.dtype]=None, **private_args_DO_NOT_USE) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Performs a fused linear op followed by a reduce-scatter

    It is equivalent to the following plain PyTorch code:

    gathered_output = torch.nn.functional.linear(gathered_input, weight)
    # like gathered_output but with first dim divided by group's world size
    scattered_output = gathered_output.new_empty(...)
    dist.reduce_scatter_tensor(scattered_output, gathered_output, group=group)

    Supports FP8 gemm with tensor-wise quantized weights. To enable FP8 gemm:
    1. pass weight and gathered_input as FP8 tensors
    2. Set `scale_gathered_input` and `scale_weight` to the scales used to quantize
    inputs and weight, respectively.
    3. Set out_dtype to the desired output dtype. If not specified, it will be inferred from
    gathered_input datatype.
    """
    world_size = group.size()
    weights = weight if isinstance(weight, list) else [weight]
    assert (scale_gathered_input is None) == (scale_weight is None)
    if scale_weight is not None:
        assert isinstance(weight, list) == isinstance(scale_weight, list)
        scales_weights = scale_weight if isinstance(scale_weight, list) else [scale_weight]
        assert len(weights) == len(scales_weights)
        assert out_dtype is not None, 'output_dtype is required with FP8'
    else:
        scales_weights = [torch.empty(1)] * len(weights)
    assert all((w.ndim == 2 for w in weights))
    assert gathered_input.ndim >= 2
    assert all((gathered_input.shape[-1] == w.shape[-1] for w in weights))
    assert gathered_input.is_contiguous()
    assert gathered_input.shape[0] % world_size == 0
    gathered_input = gathered_input.view((world_size, gathered_input.shape[0] // world_size) + gathered_input.shape[1:])
    gathered_output_shapes = [gathered_input.shape[:-1] + w.shape[:-1] for w in weights]
    scattered_output_shapes = [gos[1:] for gos in gathered_output_shapes]
    if out is not None:
        assert isinstance(out, list) == isinstance(weight, list)
        scattered_outputs = out if isinstance(out, list) else [out]
        assert len(scattered_outputs) == scattered_output_shapes
        assert all((so.device == gathered_input.device for so in scattered_outputs))
        assert all((so.dtype == gathered_input.dtype for so in scattered_outputs))
        assert all((so.shape == sos for so, sos in zip(scattered_outputs, scattered_output_shapes)))
        if out_dtype is not None:
            if isinstance(out, list):
                for o in out:
                    assert o.dtype == out_dtype
            else:
                assert out.dtype == out_dtype
    else:
        scattered_outputs = [gathered_input.new_empty(sos, dtype=out_dtype if out_dtype is not None else gathered_input.dtype) for sos in scattered_output_shapes]

    def my_matmul(outputs: List[torch.Tensor], dst_rank: int, stream_factory: Callable[[], torch.cuda.Stream]) -> None:
        for w, scale_weight, o in zip(weights, scales_weights, outputs):
            with torch.cuda.stream(stream_factory()):
                if _is_fp8_dtype(w.dtype):
                    output_amax = torch.empty(1, dtype=torch.float32, device=o.device)
                    torch._scaled_mm(gathered_input[dst_rank], w.t(), out_dtype=o.dtype, scale_a=scale_gathered_input, scale_b=scale_weight, out=(o, output_amax))
                else:
                    torch.matmul(gathered_input[dst_rank], w.t(), out=o)
    _is_regular_matmul = all([not _is_fp8_dtype(w.dtype) for w in weights])
    fused_anything_and_reducescatter(my_matmul, scattered_outputs, group=group, num_stripes=num_stripes, timeout_s=timeout_s, _is_regular_matmul=_is_regular_matmul, _extra_triton_args=dict(a_my_shard=None, a=gathered_input.flatten(0, -2), bs=[w.t() for w in weights]), **private_args_DO_NOT_USE)
    if isinstance(weight, list):
        return scattered_outputs
    else:
        return scattered_outputs[0]
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, overload
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
from .. import _is_triton_available
from .common import BaseOperator, get_xformers_operator, register_operator
from .ipc import init_ipc
def linear_and_reducescatter(self, my_matmul: Callable[[List[torch.Tensor], int, Callable[[], torch.cuda.Stream]], None], gathered_outputs: List[torch.Tensor], scattered_outputs: List[torch.Tensor], timeout_s: int, _wait: bool=True, _memcpy: bool=True, _triton: bool=True, _is_regular_matmul: bool=False, _extra_triton_args: Mapping[str, Any]={}):
    """Perform a fused linear layer followed by a reduce-scatter"""
    assert all((go.device == self.my_device for go in gathered_outputs))
    assert all((go.dtype == self.dtype for go in gathered_outputs))
    assert all((so.device == self.my_device for so in scattered_outputs))
    assert all((so.dtype == self.dtype for so in scattered_outputs))
    scattered_output_numels = [so.numel() for so in scattered_outputs]
    total_scattered_output_numel = sum(scattered_output_numels)
    self._ensure_staging_is_large_enough(total_scattered_output_numel, random_init=_memcpy is False)
    stripe = self.next_stripe % self.num_stripes
    self.next_stripe += 1
    seq_num = self.next_seq_nums[stripe] % SEQ_NUM_WRAP_AROUND
    prev_seq_num = (seq_num - 1) % SEQ_NUM_WRAP_AROUND
    self.next_seq_nums[stripe] += 1
    stagings = [s.view((self.world_size,) + so.shape) for s, so in zip(self.staging[stripe, :, :total_scattered_output_numel].split(scattered_output_numels, dim=-1), scattered_outputs)]
    buddys_stagings = [[bs] * len(scattered_outputs) if bs.numel() == 0 else [s.view(so.shape) for s, so in zip(bs[stripe, :total_scattered_output_numel].split(scattered_output_numels, dim=-1), scattered_outputs)] for bs in self.buddys_staging]
    current_stream = torch.cuda.current_stream()
    self.wait_stream.wait_stream(current_stream)
    if _wait:
        WaitValues.OPERATOR([self.num_reads_from_my_staging[(self.my_rank + iter_) % self.world_size][stripe] for iter_ in range(1, self.world_size)], prev_seq_num, current_stream, timeout_s)
    if _is_regular_matmul and self._should_use_triton(_triton):
        _launch_triton_matmul(cs=[s.flatten(0, -2) for s in stagings], cs_my_shard=[go[self.my_rank].flatten(0, -2) for go in gathered_outputs], my_rank=self.my_rank, world_size=self.world_size, wait_counters=None, write_counters=self.num_writes_into_my_staging, direction=FORWARDS_WITH_ME_LAST, stripe=stripe, seq_num=seq_num, num_stripes=self.num_stripes, timeout_s=timeout_s, _wait=_wait, **_extra_triton_args)
    else:
        self.second_stream.wait_stream(current_stream)
        stream_factory = self.make_stream_factory(current_stream)
        for iter_ in range(1, self.world_size):
            dst_rank = (self.my_rank + iter_) % self.world_size
            my_matmul([s[dst_rank] for s in stagings], dst_rank, stream_factory)
            if _wait:
                self.write_stream.wait_stream(current_stream)
                self.write_stream.wait_stream(self.second_stream)
                WriteValues.OPERATOR([self.num_writes_into_my_staging[dst_rank, stripe]], seq_num, self.write_stream)
        my_matmul([o[self.my_rank] for o in gathered_outputs], self.my_rank, stream_factory)
        current_stream.wait_stream(self.second_stream)
    for iter_ in range(1, self.world_size):
        src_rank = (self.my_rank - iter_) % self.world_size
        if _wait:
            WaitValues.OPERATOR([self.num_writes_into_buddys_staging[src_rank][stripe]], seq_num, self.wait_stream, timeout_s)
        self.memcpy_stream.wait_stream(self.wait_stream)
        if _memcpy:
            with torch.cuda.stream(self.memcpy_stream):
                for go, bs in zip(gathered_outputs, buddys_stagings[src_rank]):
                    go[src_rank].copy_(bs)
    current_stream.wait_stream(self.memcpy_stream)
    for go, so in zip(gathered_outputs, scattered_outputs):
        torch.sum(go, dim=0, out=so)
    self.write_stream.wait_stream(current_stream)
    if _wait:
        WriteValues.OPERATOR([self.num_reads_from_buddys_staging[(self.my_rank - iter_) % self.world_size, stripe] for iter_ in range(1, self.world_size)], seq_num, self.write_stream)
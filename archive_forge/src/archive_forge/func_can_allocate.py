import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
    seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
    num_required_blocks = len(seq.logical_token_blocks)
    if seq_group.prefix is not None and seq_group.prefix.allocated:
        num_required_blocks -= seq_group.prefix.get_num_blocks()
    if self.block_sliding_window is not None:
        num_required_blocks = min(num_required_blocks, self.block_sliding_window)
    num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
    if self.num_total_gpu_blocks - num_required_blocks < self.watermark_blocks:
        return AllocStatus.NEVER
    if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
        return AllocStatus.OK
    else:
        return AllocStatus.LATER
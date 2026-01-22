import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def get_num_free_gpu_blocks(self) -> int:
    return self.gpu_allocator.get_num_free_blocks()
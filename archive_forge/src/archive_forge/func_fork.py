import enum
from typing import Dict, List, Optional, Set, Tuple
from vllm.block import BlockTable, PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
    src_block_table = self.block_tables[parent_seq.seq_id]
    self.block_tables[child_seq.seq_id] = src_block_table.copy()
    for block in src_block_table:
        block.ref_count += 1
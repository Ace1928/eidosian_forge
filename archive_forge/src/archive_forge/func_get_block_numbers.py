from typing import Dict, List, Sequence, Tuple, Optional
from vllm.block import BlockTable
def get_block_numbers(self) -> List[int]:
    return [block.block_number for block in self.block_table]
from typing import Dict, List, Sequence, Tuple, Optional
from vllm.block import BlockTable
def set_block_table(self, block_table: BlockTable) -> None:
    self.block_table = block_table.copy()
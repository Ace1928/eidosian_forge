from typing import List
from vllm.utils import Device
def is_full(self) -> bool:
    return self.num_tokens == self.block_size
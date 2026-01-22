from typing import List
from vllm.utils import Device
def get_num_empty_slots(self) -> int:
    return self.block_size - self.num_tokens
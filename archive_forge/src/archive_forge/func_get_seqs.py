import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
def get_seqs(self, status: Optional[SequenceStatus]=None) -> List[Sequence]:
    if status is None:
        return list(self.seqs_dict.values())
    else:
        return [seq for seq in self.seqs_dict.values() if seq.status == status]
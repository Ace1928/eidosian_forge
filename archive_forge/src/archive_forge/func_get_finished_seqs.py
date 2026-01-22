import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
def get_finished_seqs(self) -> List[Sequence]:
    return [seq for seq in self.seqs_dict.values() if seq.is_finished()]
import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
def set_finished_time(self, time: Optional[float]) -> None:
    """Sets the finished time for Request level timings."""
    self.metrics.finished_time = time
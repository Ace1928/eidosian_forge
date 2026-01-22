import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
@staticmethod
def get_finished_reason(status: 'SequenceStatus') -> Union[str, None]:
    if status == SequenceStatus.FINISHED_STOPPED:
        finish_reason = 'stop'
    elif status == SequenceStatus.FINISHED_LENGTH_CAPPED:
        finish_reason = 'length'
    elif status == SequenceStatus.FINISHED_ABORTED:
        finish_reason = 'abort'
    elif status == SequenceStatus.FINISHED_IGNORED:
        finish_reason = 'length'
    else:
        finish_reason = None
    return finish_reason
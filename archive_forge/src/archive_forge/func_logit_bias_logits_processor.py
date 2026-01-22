import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams
import torch
def logit_bias_logits_processor(token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
    for token_id, bias in self.logit_bias.items():
        bias = min(100, max(-100, bias))
        logits[int(token_id)] += bias
    return logits
import inspect
import math
import os
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
from ..core import (
from ..import_utils import is_npu_available, is_torch_greater_2_0, is_xpu_available
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig, RunningMoments
from transformers import pipeline
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
    if self.is_encoder_decoder:
        input_data = self.data_collator([{'input_ids': q, 'attention_mask': torch.ones_like(q)} for q in queries]).to(self.current_device)
        decoder_inputs = self.data_collator([{'input_ids': r, 'attention_mask': torch.ones_like(r)} for r in responses]).to(self.current_device)
        input_data['decoder_input_ids'] = decoder_inputs['input_ids']
        input_data['decoder_attention_mask'] = decoder_inputs['attention_mask']
    else:
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
        input_data = self.data_collator([{'input_ids': ids, 'attention_mask': torch.ones_like(ids)} for ids in input_ids]).to(self.current_device)
    input_data.pop('labels', None)
    return input_data
import collections
import copy
import functools
import gc
import importlib.metadata
import inspect
import itertools
import json
import os
import re
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zipfile import is_zipfile
import torch
from packaging import version
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Identity
from torch.utils.checkpoint import checkpoint
from .activations import get_activation
from .configuration_utils import PretrainedConfig
from .dynamic_module_utils import custom_object_save
from .generation import GenerationConfig, GenerationMixin
from .integrations import PeftAdapterMixin, deepspeed_config, is_deepspeed_zero3_enabled
from .pytorch_utils import (  # noqa: F401
from .quantizers import AutoHfQuantizer, HfQuantizer
from .safetensors_conversion import auto_conversion
from .utils import (
from .utils.hub import convert_file_size_to_int, create_and_tag_model_card, get_checkpoint_shard_files
from .utils.import_utils import (
from .utils.quantization_config import BitsAndBytesConfig, QuantizationMethod
def warn_if_padding_and_no_attention_mask(self, input_ids, attention_mask):
    """
        Shows a one-time warning if the input_ids appear to contain padding and no attention mask was given.
        """
    if is_torch_fx_proxy(input_ids) or torch.jit.is_tracing() or is_torchdynamo_compiling():
        return
    if attention_mask is not None or self.config.pad_token_id is None:
        return
    if self.config.pad_token_id in input_ids[:, [-1, 0]]:
        warn_string = 'We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.'
        if self.config.bos_token_id is not None and self.config.bos_token_id == self.config.pad_token_id or (self.config.eos_token_id is not None and self.config.eos_token_id == self.config.pad_token_id) or (self.config.sep_token_id is not None and self.config.sep_token_id == self.config.pad_token_id):
            warn_string += f'\nYou may ignore this warning if your `pad_token_id` ({self.config.pad_token_id}) is identical to the `bos_token_id` ({self.config.bos_token_id}), `eos_token_id` ({self.config.eos_token_id}), or the `sep_token_id` ({self.config.sep_token_id}), and your input is not padded.'
        logger.warning_once(warn_string)
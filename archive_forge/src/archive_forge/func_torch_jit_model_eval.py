import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .integrations import (
import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
from .trainer_pt_utils import (
from .trainer_utils import (
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
from .utils.quantization_config import QuantizationMethod
def torch_jit_model_eval(self, model, dataloader, training=False):
    if not training:
        if dataloader is None:
            logger.warning('failed to use PyTorch jit mode due to current dataloader is none.')
            return model
        example_batch = next(iter(dataloader))
        example_batch = self._prepare_inputs(example_batch)
        try:
            jit_model = copy.copy(model)
            jit_model.eval()
            original_forward = jit_model.__dict__.pop('_original_forward', None)
            if original_forward:
                jit_model.forward = original_forward
            with self.accelerator.autocast(cache_enabled=False), torch.no_grad():
                if version.parse(version.parse(torch.__version__).base_version) >= version.parse('2.0.0'):
                    if isinstance(example_batch, dict):
                        jit_model = torch.jit.trace(jit_model, example_kwarg_inputs=example_batch, strict=False)
                    else:
                        jit_model = torch.jit.trace(jit_model, example_kwarg_inputs={key: example_batch[key] for key in example_batch}, strict=False)
                else:
                    jit_inputs = []
                    for key in example_batch:
                        example_tensor = torch.ones_like(example_batch[key])
                        jit_inputs.append(example_tensor)
                    jit_inputs = tuple(jit_inputs)
                    jit_model = torch.jit.trace(jit_model, jit_inputs, strict=False)
            jit_model = torch.jit.freeze(jit_model)
            with torch.no_grad():
                jit_model(**example_batch)
                jit_model(**example_batch)
            model = jit_model
            self.use_cpu_amp = False
        except (RuntimeError, TypeError, ValueError, NameError, IndexError) as e:
            logger.warning(f'failed to use PyTorch jit mode due to: {e}.')
    return model
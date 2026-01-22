import contextlib
from typing import Type
import torch
import torch.nn as nn
from vllm.config import DeviceConfig, ModelConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.weight_utils import (get_quant_config,
Sets the default torch dtype to the given dtype.
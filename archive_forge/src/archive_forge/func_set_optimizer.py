import contextlib
import io
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from huggingface_hub import get_full_repo_name
from packaging import version
from .debug_utils import DebugOption
from .trainer_utils import (
from .utils import (
from .utils.generic import strtobool
from .utils.import_utils import is_optimum_neuron_available
def set_optimizer(self, name: Union[str, OptimizerNames]='adamw_torch', learning_rate: float=5e-05, weight_decay: float=0, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-08, args: Optional[str]=None):
    """
        A method that regroups all arguments linked to the optimizer and its hyperparameters.

        Args:
            name (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_torch"`):
                The optimizer to use: `"adamw_hf"`, `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_apex_fused"`,
                `"adamw_anyprecision"` or `"adafactor"`.
            learning_rate (`float`, *optional*, defaults to 5e-5):
                The initial learning rate.
            weight_decay (`float`, *optional*, defaults to 0):
                The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights.
            beta1 (`float`, *optional*, defaults to 0.9):
                The beta1 hyperparameter for the adam optimizer or its variants.
            beta2 (`float`, *optional*, defaults to 0.999):
                The beta2 hyperparameter for the adam optimizer or its variants.
            epsilon (`float`, *optional*, defaults to 1e-8):
                The epsilon hyperparameter for the adam optimizer or its variants.
            args (`str`, *optional*):
                Optional arguments that are supplied to AnyPrecisionAdamW (only useful when
                `optim="adamw_anyprecision"`).

        Example:

        ```py
        >>> from transformers import TrainingArguments

        >>> args = TrainingArguments("working_dir")
        >>> args = args.set_optimizer(name="adamw_torch", beta1=0.8)
        >>> args.optim
        'adamw_torch'
        ```
        """
    self.optim = OptimizerNames(name)
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay
    self.adam_beta1 = beta1
    self.adam_beta2 = beta2
    self.adam_epsilon = epsilon
    self.optim_args = args
    return self
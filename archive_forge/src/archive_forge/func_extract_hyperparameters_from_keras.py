import copy
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests
import yaml
from huggingface_hub import model_info
from huggingface_hub.utils import HFValidationError
from . import __version__
from .models.auto.modeling_auto import (
from .training_args import ParallelMode
from .utils import (
def extract_hyperparameters_from_keras(model):
    from .modeling_tf_utils import keras
    hyperparameters = {}
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        hyperparameters['optimizer'] = model.optimizer.get_config()
    else:
        hyperparameters['optimizer'] = None
    hyperparameters['training_precision'] = keras.mixed_precision.global_policy().name
    return hyperparameters
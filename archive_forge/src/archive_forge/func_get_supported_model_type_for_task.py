import importlib
import inspect
import itertools
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import huggingface_hub
from packaging import version
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import SAFE_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from ..utils import CONFIG_NAME
from ..utils.import_utils import is_onnx_available
@staticmethod
def get_supported_model_type_for_task(task: str, exporter: str) -> List[str]:
    """
        Returns the list of supported architectures by the exporter for a given task. Transformers-specific.
        """
    return [model_type.replace('-', '_') for model_type in TasksManager._SUPPORTED_MODEL_TYPE if task in TasksManager._SUPPORTED_MODEL_TYPE[model_type][exporter]]
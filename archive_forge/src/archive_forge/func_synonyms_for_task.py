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
def synonyms_for_task(task: str) -> Set[str]:
    synonyms = [k for k, v in TasksManager._SYNONYM_TASK_MAP.items() if v == task]
    synonyms += [k for k, v in TasksManager._SYNONYM_TASK_MAP.items() if v == TasksManager.map_from_synonym(task)]
    synonyms = set(synonyms)
    try:
        synonyms.remove(task)
    except KeyError:
        pass
    return synonyms
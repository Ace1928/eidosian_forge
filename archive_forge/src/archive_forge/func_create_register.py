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
@classmethod
def create_register(cls, backend: str, overwrite_existing: bool=False) -> Callable[[str, Tuple[str, ...]], Callable[[Type], Type]]:
    """
        Creates a register function for the specified backend.

        Args:
            backend (`str`):
                The name of the backend that the register function will handle.
            overwrite_existing (`bool`, defaults to `False`):
                Whether or not the register function is allowed to overwrite an already existing config.

        Returns:
            `Callable[[str, Tuple[str, ...]], Callable[[Type], Type]]`: A decorator taking the model type and a the
            supported tasks.

        Example:
            ```python
            >>> register_for_new_backend = create_register("new-backend")

            >>> @register_for_new_backend("bert", "text-classification", "token-classification")
            >>> class BertNewBackendConfig(NewBackendConfig):
            >>>     pass
            ```
        """

    def wrapper(model_type: str, *supported_tasks: str, library_name: str='transformers') -> Callable[[Type], Type]:

        def decorator(config_cls: Type) -> Type:
            supported_model_type_for_library = TasksManager._LIBRARY_TO_SUPPORTED_MODEL_TYPES[library_name]
            mapping = supported_model_type_for_library.get(model_type, {})
            mapping_backend = mapping.get(backend, {})
            for task in supported_tasks:
                if task not in cls.get_all_tasks():
                    known_tasks = ', '.join(cls.get_all_tasks())
                    raise ValueError(f'The TasksManager does not know the task called "{task}", known tasks: {known_tasks}.')
                if not overwrite_existing and task in mapping_backend:
                    continue
                mapping_backend[task] = make_backend_config_constructor_for_task(config_cls, task)
            mapping[backend] = mapping_backend
            supported_model_type_for_library[model_type] = mapping
            return config_cls
        return decorator
    return wrapper
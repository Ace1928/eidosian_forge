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
def infer_library_from_model(cls, model_name_or_path: Union[str, Path], subfolder: str='', revision: Optional[str]=None, cache_dir: str=huggingface_hub.constants.HUGGINGFACE_HUB_CACHE, library_name: Optional[str]=None):
    """
        Infers the library from the model repo.

        Args:
            model_name_or_path (`str`):
                The model to infer the task from. This can either be the name of a repo on the HuggingFace Hub, an
                instance of a model, or a model class.
            subfolder (`str`, defaults to `""`):
                In case the model files are located inside a subfolder of the model directory / repo on the Hugging
                Face Hub, you can specify the subfolder name here.
            revision (`Optional[str]`, *optional*, defaults to `None`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, *optional*):
                Path to a directory in which a downloaded pretrained model weights have been cached if the standard cache should not be used.
            library_name (`Optional[str]`, *optional*):
                The library name of the model. Can be any of "transformers", "timm", "diffusers", "sentence_transformers".
        Returns:
            `str`: The library name automatically detected from the model repo.
        """
    if library_name is not None:
        return library_name
    all_files, _ = TasksManager.get_model_files(model_name_or_path, subfolder, cache_dir)
    if 'model_index.json' in all_files:
        library_name = 'diffusers'
    elif any((file_path.startswith('sentence_') for file_path in all_files)) or 'config_sentence_transformers.json' in all_files:
        library_name = 'sentence_transformers'
    elif CONFIG_NAME in all_files:
        kwargs = {'subfolder': subfolder, 'revision': revision, 'cache_dir': cache_dir}
        config_dict, kwargs = PretrainedConfig.get_config_dict(model_name_or_path, **kwargs)
        model_config = PretrainedConfig.from_dict(config_dict, **kwargs)
        if hasattr(model_config, 'pretrained_cfg') or hasattr(model_config, 'architecture'):
            library_name = 'timm'
        elif hasattr(model_config, '_diffusers_version'):
            library_name = 'diffusers'
        else:
            library_name = 'transformers'
    else:
        library_name = 'transformers'
    if library_name is None:
        raise ValueError('The library name could not be automatically inferred. If using the command-line, please provide the argument --library {transformers,diffusers,timm,sentence_transformers}. Example: `--library diffusers`.')
    return library_name
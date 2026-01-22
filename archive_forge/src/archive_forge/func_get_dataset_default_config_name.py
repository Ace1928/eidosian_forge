import inspect
import os
import shutil
import warnings
from pathlib import Path, PurePath
from typing import Dict, List, Mapping, Optional, Sequence, Union
import huggingface_hub
from . import config
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager
from .info import DatasetInfo
from .load import (
from .utils.deprecation_utils import deprecated
from .utils.file_utils import relative_to_absolute_path
from .utils.logging import get_logger
from .utils.version import Version
def get_dataset_default_config_name(path: str, revision: Optional[Union[str, Version]]=None, download_config: Optional[DownloadConfig]=None, download_mode: Optional[Union[DownloadMode, str]]=None, dynamic_modules_path: Optional[str]=None, data_files: Optional[Union[Dict, List, str]]=None, **download_kwargs) -> Optional[str]:
    """Get the default config name for a particular dataset.

    Args:
        path (`str`): path to the dataset processing script with the dataset builder. Can be either:

            - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with [`datasets.list_datasets`])
                e.g. `'squad'`, `'glue'` or `'openai/webtext'`
        revision (`Union[str, datasets.Version]`, *optional*):
            If specified, the dataset module will be loaded from the datasets repository at this version.
            By default:
            - it is set to the local version of the lib.
            - it will also try to load it from the main branch if it's not available at the local version of the lib.
            Specifying a version that is different from your local version of the lib might cause compatibility issues.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        download_mode ([`DownloadMode`] or `str`, defaults to `REUSE_DATASET_IF_EXISTS`):
            Download/generate mode.
        dynamic_modules_path (`str`, defaults to `~/.cache/huggingface/modules/datasets_modules`):
            Optional path to the directory in which the dynamic modules are saved. It must have been initialized with `init_dynamic_modules`.
            By default the datasets and metrics are stored inside the `datasets_modules` module.
        data_files (`Union[Dict, List, str]`, *optional*):
            Defining the data_files of the dataset configuration.
        **download_kwargs (additional keyword arguments):
            Optional attributes for [`DownloadConfig`] which will override the attributes in `download_config` if supplied,
            for example `token`.

    Returns:
        Optional[str]

    Example:

    ```py
    >>> from datasets import get_dataset_default_config_name
    >>> get_dataset_default_config_name("openbookqa")
    'main'
    ```
    """
    dataset_module = dataset_module_factory(path, revision=revision, download_config=download_config, download_mode=download_mode, dynamic_modules_path=dynamic_modules_path, data_files=data_files, **download_kwargs)
    builder_cls = get_dataset_builder_class(dataset_module, dataset_name=os.path.basename(path))
    builder_configs = list(builder_cls.builder_configs.keys())
    if builder_configs:
        default_config_name = builder_configs[0] if len(builder_configs) == 1 else None
    else:
        default_config_name = 'default'
    return builder_cls.DEFAULT_CONFIG_NAME or default_config_name
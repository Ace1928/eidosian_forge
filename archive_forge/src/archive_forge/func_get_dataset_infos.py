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
def get_dataset_infos(path: str, data_files: Optional[Union[Dict, List, str]]=None, download_config: Optional[DownloadConfig]=None, download_mode: Optional[Union[DownloadMode, str]]=None, revision: Optional[Union[str, Version]]=None, token: Optional[Union[bool, str]]=None, use_auth_token='deprecated', **config_kwargs):
    """Get the meta information about a dataset, returned as a dict mapping config name to DatasetInfoDict.

    Args:
        path (`str`): path to the dataset processing script with the dataset builder. Can be either:

            - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with [`datasets.list_datasets`])
                e.g. `'squad'`, `'glue'` or``'openai/webtext'`
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
        data_files (`Union[Dict, List, str]`, *optional*):
            Defining the data_files of the dataset configuration.
        token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.
        use_auth_token (`str` or `bool`, *optional*):
            Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If `True`, or not specified, will get token from `"~/.huggingface"`.

            <Deprecated version="2.14.0">

            `use_auth_token` was deprecated in favor of `token` in version 2.14.0 and will be removed in 3.0.0.

            </Deprecated>

        **config_kwargs (additional keyword arguments):
            Optional attributes for builder class which will override the attributes if supplied.

    Example:

    ```py
    >>> from datasets import get_dataset_infos
    >>> get_dataset_infos('rotten_tomatoes')
    {'default': DatasetInfo(description="Movie Review Dataset.
This is a dataset of containing 5,331 positive and 5,331 negative processed
sentences from Rotten Tomatoes movie reviews...), ...}
    ```
    """
    if use_auth_token != 'deprecated':
        warnings.warn("'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\nYou can remove this warning by passing 'token=<use_auth_token>' instead.", FutureWarning)
        token = use_auth_token
    config_names = get_dataset_config_names(path=path, revision=revision, download_config=download_config, download_mode=download_mode, data_files=data_files, token=token)
    return {config_name: get_dataset_config_info(path=path, config_name=config_name, data_files=data_files, download_config=download_config, download_mode=download_mode, revision=revision, token=token, **config_kwargs) for config_name in config_names}
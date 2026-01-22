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
@deprecated('Clone the dataset repository from the Hugging Face Hub instead.')
def inspect_dataset(path: str, local_path: str, download_config: Optional[DownloadConfig]=None, **download_kwargs):
    """
    Allow inspection/modification of a dataset script by copying on local drive at local_path.

    Args:
        path (`str`): Path to the dataset processing script with the dataset builder. Can be either:

            - a local path to processing script or the directory containing the script (if the script has the same name
                as the directory),
                e.g. `'./dataset/squad'` or `'./dataset/squad/squad.py'`.
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with [`list_datasets`])
                e.g. `'squad'`, `'glue'` or `'openai/webtext'`.
        local_path (`str`):
            Path to the local folder to copy the dataset script to.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters.
        **download_kwargs (additional keyword arguments):
            Optional arguments for [`DownloadConfig`] which will override
            the attributes of `download_config` if supplied.
    """
    if download_config is None:
        download_config = DownloadConfig(**download_kwargs)
    if os.path.isfile(path):
        path = str(Path(path).parent)
    if os.path.isdir(path):
        shutil.copytree(path, local_path, dirs_exist_ok=True)
    else:
        huggingface_hub.HfApi(endpoint=config.HF_ENDPOINT, token=download_config.token).snapshot_download(repo_id=path, repo_type='dataset', local_dir=local_path, force_download=download_config.force_download)
    print(f'The dataset {path} can be inspected at {local_path}. You can modify this loading script  if it has one and use it with `datasets.load_dataset("{PurePath(local_path).as_posix()}")`.')
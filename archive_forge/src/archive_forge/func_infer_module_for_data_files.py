import filecmp
import glob
import importlib
import inspect
import json
import os
import posixpath
import shutil
import signal
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import fsspec
import requests
import yaml
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, HfFileSystem
from . import config
from .arrow_dataset import Dataset
from .builder import BuilderConfig, DatasetBuilder
from .data_files import (
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadMode
from .download.streaming_download_manager import StreamingDownloadManager, xbasename, xglob, xjoin
from .exceptions import DataFilesNotFoundError, DatasetNotFoundError
from .features import Features
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict
from .iterable_dataset import IterableDataset
from .metric import Metric
from .naming import camelcase_to_snakecase, snakecase_to_camelcase
from .packaged_modules import (
from .splits import Split
from .utils import _datasets_server
from .utils._filelock import FileLock
from .utils.deprecation_utils import deprecated
from .utils.file_utils import (
from .utils.hub import hf_hub_url
from .utils.info_utils import VerificationMode, is_small_dataset
from .utils.logging import get_logger
from .utils.metadata import MetadataConfigs
from .utils.py_utils import get_imports
from .utils.version import Version
def infer_module_for_data_files(data_files: DataFilesDict, path: Optional[str]=None, download_config: Optional[DownloadConfig]=None) -> Tuple[Optional[str], Dict[str, Any]]:
    """Infer module (and builder kwargs) from data files. Raise if module names for different splits don't match.

    Args:
        data_files ([`DataFilesDict`]): Dict of list of data files.
        path (str, *optional*): Dataset name or path.
        download_config ([`DownloadConfig`], *optional*):
            Specific download configuration parameters to authenticate on the Hugging Face Hub for private remote files.

    Returns:
        tuple[str, dict[str, Any]]: Tuple with
            - inferred module name
            - builder kwargs
    """
    split_modules = {split: infer_module_for_data_files_list(data_files_list, download_config=download_config) for split, data_files_list in data_files.items()}
    module_name, default_builder_kwargs = next(iter(split_modules.values()))
    if any(((module_name, default_builder_kwargs) != split_module for split_module in split_modules.values())):
        raise ValueError(f"Couldn't infer the same data file format for all splits. Got {split_modules}")
    if not module_name:
        raise DataFilesNotFoundError('No (supported) data files found' + (f' in {path}' if path else ''))
    return (module_name, default_builder_kwargs)
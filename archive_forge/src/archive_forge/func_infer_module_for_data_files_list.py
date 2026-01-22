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
def infer_module_for_data_files_list(data_files_list: DataFilesList, download_config: Optional[DownloadConfig]=None) -> Tuple[Optional[str], dict]:
    """Infer module (and builder kwargs) from list of data files.

    It picks the module based on the most common file extension.
    In case of a draw ".parquet" is the favorite, and then alphabetical order.

    Args:
        data_files_list (DataFilesList): List of data files.
        download_config (bool or str, optional): mainly use use_auth_token or storage_options to support different platforms and auth types.

    Returns:
        tuple[str, dict[str, Any]]: Tuple with
            - inferred module name
            - dict of builder kwargs
    """
    extensions_counter = Counter((('.' + suffix.lower(), xbasename(filepath) in ('metadata.jsonl', 'metadata.csv')) for filepath in data_files_list[:config.DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE] for suffix in xbasename(filepath).split('.')[1:]))
    if extensions_counter:

        def sort_key(ext_count: Tuple[Tuple[str, bool], int]) -> Tuple[int, bool]:
            """Sort by count and set ".parquet" as the favorite in case of a draw, and ignore metadata files"""
            (ext, is_metadata), count = ext_count
            return (not is_metadata, count, ext == '.parquet', ext)
        for (ext, _), _ in sorted(extensions_counter.items(), key=sort_key, reverse=True):
            if ext in _EXTENSION_TO_MODULE:
                return _EXTENSION_TO_MODULE[ext]
            elif ext == '.zip':
                return infer_module_for_data_files_list_in_archives(data_files_list, download_config=download_config)
    return (None, {})
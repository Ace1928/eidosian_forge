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
def infer_module_for_data_files_list_in_archives(data_files_list: DataFilesList, download_config: Optional[DownloadConfig]=None) -> Tuple[Optional[str], dict]:
    """Infer module (and builder kwargs) from list of archive data files.

    Args:
        data_files_list (DataFilesList): List of data files.
        download_config (bool or str, optional): mainly use use_auth_token or storage_options to support different platforms and auth types.

    Returns:
        tuple[str, dict[str, Any]]: Tuple with
            - inferred module name
            - dict of builder kwargs
    """
    archived_files = []
    archive_files_counter = 0
    for filepath in data_files_list:
        if str(filepath).endswith('.zip'):
            archive_files_counter += 1
            if archive_files_counter > config.GLOBBED_DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE:
                break
            extracted = xjoin(StreamingDownloadManager().extract(filepath), '**')
            archived_files += [f.split('::')[0] for f in xglob(extracted, recursive=True, download_config=download_config)[:config.ARCHIVED_DATA_FILES_MAX_NUMBER_FOR_MODULE_INFERENCE]]
    extensions_counter = Counter(('.' + suffix.lower() for filepath in archived_files for suffix in xbasename(filepath).split('.')[1:]))
    if extensions_counter:
        most_common = extensions_counter.most_common(1)[0][0]
        if most_common in _EXTENSION_TO_MODULE:
            return _EXTENSION_TO_MODULE[most_common]
    return (None, {})
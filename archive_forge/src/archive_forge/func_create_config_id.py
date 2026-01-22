import abc
import contextlib
import copy
import inspect
import os
import posixpath
import shutil
import textwrap
import time
import urllib
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, Optional, Tuple, Union
from unittest.mock import patch
import fsspec
import pyarrow as pa
from multiprocess import Pool
from tqdm.contrib.concurrent import thread_map
from . import config, utils
from .arrow_dataset import Dataset
from .arrow_reader import (
from .arrow_writer import ArrowWriter, BeamWriter, ParquetWriter, SchemaInferenceError
from .data_files import DataFilesDict, DataFilesPatternsDict, sanitize_patterns
from .dataset_dict import DatasetDict, IterableDatasetDict
from .download.download_config import DownloadConfig
from .download.download_manager import DownloadManager, DownloadMode
from .download.mock_download_manager import MockDownloadManager
from .download.streaming_download_manager import StreamingDownloadManager, xjoin, xopen
from .exceptions import DatasetGenerationCastError, DatasetGenerationError, FileFormatError, ManualDownloadError
from .features import Features
from .filesystems import (
from .fingerprint import Hasher
from .info import DatasetInfo, DatasetInfosDict, PostProcessedInfo
from .iterable_dataset import ArrowExamplesIterable, ExamplesIterable, IterableDataset
from .keyhash import DuplicatedKeysError
from .naming import INVALID_WINDOWS_CHARACTERS_IN_PATH, camelcase_to_snakecase
from .splits import Split, SplitDict, SplitGenerator, SplitInfo
from .streaming import extend_dataset_builder_for_streaming
from .table import CastError
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils._filelock import FileLock
from .utils.file_utils import cached_path, is_remote_url
from .utils.info_utils import VerificationMode, get_size_checksum_dict, verify_checksums, verify_splits
from .utils.py_utils import (
from .utils.sharding import _number_of_shards_in_gen_kwargs, _split_gen_kwargs
from .utils.track import tracked_list
def create_config_id(self, config_kwargs: dict, custom_features: Optional[Features]=None) -> str:
    """
        The config id is used to build the cache directory.
        By default it is equal to the config name.
        However the name of a config is not sufficient to have a unique identifier for the dataset being generated
        since it doesn't take into account:
        - the config kwargs that can be used to overwrite attributes
        - the custom features used to write the dataset
        - the data_files for json/text/csv/pandas datasets

        Therefore the config id is just the config name with an optional suffix based on these.
        """
    suffix: Optional[str] = None
    config_kwargs_to_add_to_suffix = config_kwargs.copy()
    config_kwargs_to_add_to_suffix.pop('name', None)
    config_kwargs_to_add_to_suffix.pop('version', None)
    if 'data_dir' in config_kwargs_to_add_to_suffix:
        if config_kwargs_to_add_to_suffix['data_dir'] is None:
            config_kwargs_to_add_to_suffix.pop('data_dir', None)
        else:
            data_dir = config_kwargs_to_add_to_suffix['data_dir']
            data_dir = os.path.normpath(data_dir)
            config_kwargs_to_add_to_suffix['data_dir'] = data_dir
    if config_kwargs_to_add_to_suffix:
        config_kwargs_to_add_to_suffix = {k: config_kwargs_to_add_to_suffix[k] for k in sorted(config_kwargs_to_add_to_suffix)}
        if all((isinstance(v, (str, bool, int, float)) for v in config_kwargs_to_add_to_suffix.values())):
            suffix = ','.join((str(k) + '=' + urllib.parse.quote_plus(str(v)) for k, v in config_kwargs_to_add_to_suffix.items()))
            if len(suffix) > 32:
                suffix = Hasher.hash(config_kwargs_to_add_to_suffix)
        else:
            suffix = Hasher.hash(config_kwargs_to_add_to_suffix)
    if custom_features is not None:
        m = Hasher()
        if suffix:
            m.update(suffix)
        m.update(custom_features)
        suffix = m.hexdigest()
    if suffix:
        config_id = self.name + '-' + suffix
        if len(config_id) > config.MAX_DATASET_CONFIG_ID_READABLE_LENGTH:
            config_id = self.name + '-' + Hasher.hash(suffix)
        return config_id
    else:
        return self.name
import contextlib
import copy
import fnmatch
import json
import math
import posixpath
import re
import warnings
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import fsspec
import numpy as np
from huggingface_hub import (
from . import config
from .arrow_dataset import PUSH_TO_HUB_WITHOUT_METADATA_CONFIGS_SPLIT_PATTERN_SHARDED, Dataset
from .features import Features
from .features.features import FeatureType
from .info import DatasetInfo, DatasetInfosDict
from .naming import _split_re
from .splits import NamedSplit, Split, SplitDict, SplitInfo
from .table import Table
from .tasks import TaskTemplate
from .utils import logging
from .utils.deprecation_utils import deprecated
from .utils.doc_utils import is_documented_by
from .utils.hub import list_files_info
from .utils.metadata import MetadataConfigs
from .utils.py_utils import asdict, glob_pattern_to_regex, string_to_dict
from .utils.typing import PathLike
def flatten_indices(self, keep_in_memory: bool=False, cache_file_names: Optional[Dict[str, Optional[str]]]=None, writer_batch_size: Optional[int]=1000, features: Optional[Features]=None, disable_nullable: bool=False, num_proc: Optional[int]=None, new_fingerprint: Optional[str]=None) -> 'DatasetDict':
    """Create and cache a new Dataset by flattening the indices mapping.

        Args:
            keep_in_memory (`bool`, defaults to `False`):
                Keep the dataset in memory instead of writing it to a cache file.
            cache_file_names (`Dict[str, str]`, *optional*, default `None`):
                Provide the name of a path for the cache file. It is used to store the
                results of the computation instead of the automatically generated cache file name.
                You have to provide one `cache_file_name` per dataset in the dataset dictionary.
            writer_batch_size (`int`, defaults to `1000`):
                Number of rows per write operation for the cache file writer.
                This value is a good trade-off between memory usage during the processing, and processing speed.
                Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `map`.
            features (`Optional[datasets.Features]`, defaults to `None`):
                Use a specific [`Features`] to store the cache file
                instead of the automatically generated one.
            disable_nullable (`bool`, defaults to `False`):
                Allow null values in the table.
            num_proc (`int`, optional, default `None`):
                Max number of processes when generating cache. Already cached shards are loaded sequentially
            new_fingerprint (`str`, *optional*, defaults to `None`):
                The new fingerprint of the dataset after transform.
                If `None`, the new fingerprint is computed using a hash of the previous fingerprint, and the transform arguments
        """
    self._check_values_type()
    if cache_file_names is None:
        cache_file_names = {k: None for k in self}
    return DatasetDict({k: dataset.flatten_indices(keep_in_memory=keep_in_memory, cache_file_name=cache_file_names[k], writer_batch_size=writer_batch_size, features=features, disable_nullable=disable_nullable, num_proc=num_proc, new_fingerprint=new_fingerprint) for k, dataset in self.items()})
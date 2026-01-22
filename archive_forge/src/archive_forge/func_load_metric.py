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
@deprecated("Use 'evaluate.load' instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate")
def load_metric(path: str, config_name: Optional[str]=None, process_id: int=0, num_process: int=1, cache_dir: Optional[str]=None, experiment_id: Optional[str]=None, keep_in_memory: bool=False, download_config: Optional[DownloadConfig]=None, download_mode: Optional[Union[DownloadMode, str]]=None, revision: Optional[Union[str, Version]]=None, trust_remote_code: Optional[bool]=None, **metric_init_kwargs) -> Metric:
    """Load a `datasets.Metric`.

    <Deprecated version="2.5.0">

    Use `evaluate.load` instead, from the new library 🤗 Evaluate: https://huggingface.co/docs/evaluate

    </Deprecated>

    Args:

        path (``str``):
            path to the metric processing script with the metric builder. Can be either:
                - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                    e.g. ``'./metrics/rouge'`` or ``'./metrics/rogue/rouge.py'``
                - a metric identifier on the HuggingFace datasets repo (list all available metrics with ``datasets.list_metrics()``)
                    e.g. ``'rouge'`` or ``'bleu'``
        config_name (:obj:`str`, optional): selecting a configuration for the metric (e.g. the GLUE metric has a configuration for each subset)
        process_id (:obj:`int`, optional): for distributed evaluation: id of the process
        num_process (:obj:`int`, optional): for distributed evaluation: total number of processes
        cache_dir (Optional str): path to store the temporary predictions and references (default to `~/.cache/huggingface/metrics/`)
        experiment_id (``str``): A specific experiment id. This is used if several distributed evaluations share the same file system.
            This is useful to compute metrics in distributed setups (in particular non-additive metrics like F1).
        keep_in_memory (bool): Whether to store the temporary results in memory (defaults to False)
        download_config (Optional ``datasets.DownloadConfig``: specific download configuration parameters.
        download_mode (:class:`DownloadMode` or :obj:`str`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        revision (Optional ``Union[str, datasets.Version]``): if specified, the module will be loaded from the datasets repository
            at this version. By default, it is set to the local version of the lib. Specifying a version that is different from
            your local version of the lib might cause compatibility issues.
        trust_remote_code (`bool`, defaults to `True`):
            Whether or not to allow for datasets defined on the Hub using a dataset script. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

            <Tip warning={true}>

            `trust_remote_code` will default to False in the next major release.

            </Tip>

            <Added version="2.16.0"/>

    Returns:
        `datasets.Metric`

    Example:

    ```py
    >>> from datasets import load_metric
    >>> accuracy = load_metric('accuracy')
    >>> accuracy.compute(references=[1, 0], predictions=[1, 1])
    {'accuracy': 0.5}
    ```
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*https://huggingface.co/docs/evaluate$', category=FutureWarning)
        download_mode = DownloadMode(download_mode or DownloadMode.REUSE_DATASET_IF_EXISTS)
        metric_module = metric_module_factory(path, revision=revision, download_config=download_config, download_mode=download_mode, trust_remote_code=trust_remote_code).module_path
        metric_cls = import_main_class(metric_module, dataset=False)
        metric = metric_cls(config_name=config_name, process_id=process_id, num_process=num_process, cache_dir=cache_dir, keep_in_memory=keep_in_memory, experiment_id=experiment_id, **metric_init_kwargs)
        metric.download_and_prepare(download_config=download_config)
        return metric
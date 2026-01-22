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
def get_dataset_config_info(path: str, config_name: Optional[str]=None, data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]]=None, download_config: Optional[DownloadConfig]=None, download_mode: Optional[Union[DownloadMode, str]]=None, revision: Optional[Union[str, Version]]=None, token: Optional[Union[bool, str]]=None, use_auth_token='deprecated', **config_kwargs) -> DatasetInfo:
    """Get the meta information (DatasetInfo) about a dataset for a particular config

    Args:
        path (``str``): path to the dataset processing script with the dataset builder. Can be either:

            - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'``
            - a dataset identifier on the Hugging Face Hub (list all available datasets and ids with ``datasets.list_datasets()``)
                e.g. ``'squad'``, ``'glue'`` or ``'openai/webtext'``
        config_name (:obj:`str`, optional): Defining the name of the dataset configuration.
        data_files (:obj:`str` or :obj:`Sequence` or :obj:`Mapping`, optional): Path(s) to source data file(s).
        download_config (:class:`~download.DownloadConfig`, optional): Specific download configuration parameters.
        download_mode (:class:`DownloadMode` or :obj:`str`, default ``REUSE_DATASET_IF_EXISTS``): Download/generate mode.
        revision (:class:`~utils.Version` or :obj:`str`, optional): Version of the dataset script to load.
            As datasets have their own git repository on the Datasets Hub, the default version "main" corresponds to their "main" branch.
            You can specify a different version than the default "main" by using a commit SHA or a git tag of the dataset repository.
        token (``str`` or :obj:`bool`, optional): Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If True, or not specified, will get token from `"~/.huggingface"`.
        use_auth_token (``str`` or :obj:`bool`, optional): Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If True, or not specified, will get token from `"~/.huggingface"`.

            <Deprecated version="2.14.0">

            `use_auth_token` was deprecated in favor of `token` in version 2.14.0 and will be removed in 3.0.0.

            </Deprecated>

        **config_kwargs (additional keyword arguments): optional attributes for builder class which will override the attributes if supplied.

    """
    if use_auth_token != 'deprecated':
        warnings.warn("'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.\nYou can remove this warning by passing 'token=<use_auth_token>' instead.", FutureWarning)
        token = use_auth_token
    builder = load_dataset_builder(path, name=config_name, data_files=data_files, download_config=download_config, download_mode=download_mode, revision=revision, token=token, **config_kwargs)
    info = builder.info
    if info.splits is None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        if token is not None:
            download_config.token = token
        builder._check_manual_download(StreamingDownloadManager(base_path=builder.base_path, download_config=download_config))
        try:
            info.splits = {split_generator.name: {'name': split_generator.name, 'dataset_name': path} for split_generator in builder._split_generators(StreamingDownloadManager(base_path=builder.base_path, download_config=download_config))}
        except Exception as err:
            raise SplitsNotFoundError('The split names could not be parsed from the dataset config.') from err
    return info
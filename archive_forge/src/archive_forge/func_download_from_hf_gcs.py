import copy
import math
import os
import re
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.contrib.concurrent import thread_map
from .download.download_config import DownloadConfig
from .naming import _split_re, filenames_for_dataset_split
from .table import InMemoryTable, MemoryMappedTable, Table, concat_tables
from .utils import logging
from .utils import tqdm as hf_tqdm
from .utils.file_utils import cached_path
def download_from_hf_gcs(self, download_config: DownloadConfig, relative_data_dir):
    """
        Download the dataset files from the Hf GCS

        Args:
            dl_cache_dir: `str`, the local cache directory used to download files
            relative_data_dir: `str`, the relative directory of the remote files from
                the `datasets` directory on GCS.

        """
    remote_cache_dir = HF_GCP_BASE_URL + '/' + relative_data_dir.replace(os.sep, '/')
    try:
        remote_dataset_info = os.path.join(remote_cache_dir, 'dataset_info.json')
        downloaded_dataset_info = cached_path(remote_dataset_info.replace(os.sep, '/'), download_config=download_config)
        shutil.move(downloaded_dataset_info, os.path.join(self._path, 'dataset_info.json'))
        if self._info is not None:
            self._info.update(self._info.from_directory(self._path))
    except FileNotFoundError as err:
        raise DatasetNotOnHfGcsError(err) from None
    try:
        for split in self._info.splits:
            file_instructions = self.get_file_instructions(name=self._info.builder_name, instruction=split, split_infos=self._info.splits.values())
            for file_instruction in file_instructions:
                file_to_download = str(Path(file_instruction['filename']).relative_to(self._path))
                remote_prepared_filename = os.path.join(remote_cache_dir, file_to_download)
                downloaded_prepared_filename = cached_path(remote_prepared_filename.replace(os.sep, '/'), download_config=download_config)
                shutil.move(downloaded_prepared_filename, file_instruction['filename'])
    except FileNotFoundError as err:
        raise MissingFilesOnHfGcsError(err) from None
import json
import os
import re
from typing import TYPE_CHECKING
import fsspec
import numpy as np
import pandas
import pandas._libs.lib as lib
from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile
from packaging import version
from pandas.io.common import stringify_path
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
@property
def row_groups_per_file(self):
    from fastparquet import ParquetFile
    if self._row_groups_per_file is None:
        row_groups_per_file = []
        for file in self.files:
            with self.fs.open(file) as f:
                row_groups = ParquetFile(f).info['row_groups']
                row_groups_per_file.append(row_groups)
        self._row_groups_per_file = row_groups_per_file
    return self._row_groups_per_file
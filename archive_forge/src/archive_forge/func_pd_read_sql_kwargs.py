import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import pandas as pd
import pyarrow as pa
import datasets
import datasets.config
from datasets.features.features import require_storage_cast
from datasets.table import table_cast
@property
def pd_read_sql_kwargs(self):
    pd_read_sql_kwargs = {'index_col': self.index_col, 'columns': self.columns, 'params': self.params, 'coerce_float': self.coerce_float, 'parse_dates': self.parse_dates}
    return pd_read_sql_kwargs
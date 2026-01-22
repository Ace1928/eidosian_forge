import copy
import os
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, TypeVar, Union
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.types
from . import config
from .utils.logging import get_logger
def table_cast(table: pa.Table, schema: pa.Schema):
    """Improved version of `pa.Table.cast`.

    It supports casting to feature types stored in the schema metadata.

    Args:
        table (`pyarrow.Table`):
            PyArrow table to cast.
        schema (`pyarrow.Schema`):
            Target PyArrow schema.

    Returns:
        table (`pyarrow.Table`): the casted table
    """
    if table.schema != schema:
        return cast_table_to_schema(table, schema)
    elif table.schema.metadata != schema.metadata:
        return table.replace_schema_metadata(schema.metadata)
    else:
        return table
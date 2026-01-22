import pickle
from typing import Any, Iterable, List, Tuple, Optional
import pandas as pd
import pyarrow as pa
import pyspark
import pyspark.sql as ps
import pyspark.sql.types as pt
from packaging import version
from pyarrow.types import is_list, is_struct, is_timestamp
from pyspark.sql.pandas.types import (
from triad.collections import Schema
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP, cast_pa_table
from triad.utils.schema import quote_name
import fugue.api as fa
from fugue import DataFrame
from .misc import is_spark_dataframe
def to_select_expression(schema_from: Any, schema_to: Any) -> List[str]:
    schema1 = to_spark_schema(schema_from)
    if isinstance(schema_to, List):
        return [schema1[n].name for n in schema_to]
    schema2 = to_spark_schema(schema_to)
    sub = pt.StructType([schema1[x.name] for x in schema2.fields])
    _, expr = to_cast_expression(sub, schema2, allow_name_mismatch=False)
    return expr
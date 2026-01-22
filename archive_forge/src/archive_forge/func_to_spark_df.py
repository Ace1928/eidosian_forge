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
def to_spark_df(session: ps.SparkSession, df: Any, schema: Any=None) -> ps.DataFrame:
    if schema is not None and (not isinstance(schema, pt.StructType)):
        schema = to_spark_schema(schema)
    if isinstance(df, pd.DataFrame):
        if pd.__version__ >= '2' and session.version < '3.4':
            if schema is None:
                schema = to_spark_schema(fa.get_schema(df))
            df = fa.as_fugue_df(df).as_array(type_safe=True)
        return pd_to_spark_df(session, df, schema=schema)
    if isinstance(df, DataFrame):
        if pd.__version__ >= '2' and session.version < '3.4':
            if schema is None:
                schema = to_spark_schema(df.schema)
            return session.createDataFrame(df.as_array(type_safe=True), schema=schema)
        return pd_to_spark_df(session, df.as_pandas(), schema=schema)
    else:
        return session.createDataFrame(df, schema=schema)
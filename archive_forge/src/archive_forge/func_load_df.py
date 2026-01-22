from typing import Any, Callable, Dict, List, Optional, Union
import pyspark.sql as ps
from pyspark.sql import SparkSession
from triad.collections import Schema
from triad.collections.dict import ParamDict
from triad.utils.assertion import assert_or_throw
from fugue._utils.io import FileParser, save_df
from fugue.collections.partition import PartitionSpec
from fugue.dataframe import DataFrame, PandasDataFrame
from fugue_spark.dataframe import SparkDataFrame
from .convert import to_schema, to_spark_schema
def load_df(self, uri: Union[str, List[str]], format_hint: Optional[str]=None, columns: Any=None, **kwargs: Any) -> DataFrame:
    if isinstance(uri, str):
        fp = [FileParser(uri, format_hint)]
    else:
        fp = [FileParser(u, format_hint) for u in uri]
    fmts = list(set((f.file_format for f in fp)))
    assert_or_throw(len(fmts) == 1, NotImplementedError("can't support multiple formats"))
    fmt = fmts[0]
    files = [f.path for f in fp]
    return self._loads[fmt](files, columns, **kwargs)
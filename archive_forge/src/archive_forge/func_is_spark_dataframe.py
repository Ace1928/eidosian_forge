from typing import Any
import pyspark.sql as ps
def is_spark_dataframe(df: Any) -> bool:
    return isinstance(df, ps.DataFrame) or (SparkConnectDataFrame is not None and isinstance(df, SparkConnectDataFrame))
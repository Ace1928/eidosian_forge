from typing import Any
import pyspark.sql as ps
def is_spark_connect(session: Any) -> bool:
    return SparkConnectSession is not None and isinstance(session, (SparkConnectSession, SparkConnectDataFrame))
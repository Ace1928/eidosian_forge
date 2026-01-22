from typing import Any
import pyspark.sql as ps
def is_spark_session(session: Any) -> bool:
    return isinstance(session, ps.SparkSession) or (SparkConnectSession is not None and isinstance(session, SparkConnectSession))
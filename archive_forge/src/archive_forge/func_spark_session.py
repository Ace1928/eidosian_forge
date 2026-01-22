from __future__ import annotations
import decimal
import signal
import sys
import threading
import pytest
from dask.datasets import timeseries
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_150, PANDAS_GE_200
from dask.dataframe.utils import assert_eq
@pytest.fixture(scope='module')
def spark_session():
    prev = signal.getsignal(signal.SIGINT)
    spark = pyspark.sql.SparkSession.builder.master('local').appName('Dask Testing').config('spark.sql.session.timeZone', 'UTC').getOrCreate()
    yield spark
    spark.stop()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, prev)
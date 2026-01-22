from collections import OrderedDict
from datetime import date, time
import numpy as np
import pandas as pd
import pyarrow as pa

    Dataframe with list columns of every possible primitive type.

    Returns
    -------
    df: pandas.DataFrame
    schema: pyarrow.Schema
        Arrow schema definition that is in line with the constructed df.
    parquet_compatible: bool
        Exclude types not supported by parquet
    
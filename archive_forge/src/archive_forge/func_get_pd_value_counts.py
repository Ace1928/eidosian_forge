from collections import Counter, OrderedDict
from functools import partial
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pandas.api.types
from ray.air.util.data_batch_conversion import BatchFormat
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor, PreprocessorNotFittedException
from ray.util.annotations import PublicAPI
def get_pd_value_counts(df: pd.DataFrame) -> List[Dict[str, Counter]]:
    df_columns = df.columns.tolist()
    result = {}
    for col in columns:
        if col in df_columns:
            result[col] = [get_pd_value_counts_per_column(df[col])]
        else:
            raise ValueError(f"Column '{col}' does not exist in DataFrame, which has columns: {df_columns}")
    return result
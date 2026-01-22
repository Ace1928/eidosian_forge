import math
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.utils.convert import to_size
from .iter import slice_iterable
def get_keys_ndarray(self, batch: pa.Table, keys: List[str]) -> np.ndarray:
    return batch.select(keys).to_pandas().to_numpy()
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
def encode_list(element: list, *, name: str):
    if isinstance(element, np.ndarray):
        element = element.tolist()
    elif not isinstance(element, list):
        element = [element]
    stats = self.stats_[f'unique_values({name})']
    counter = Counter(element)
    return [counter.get(x, 0) for x in stats]
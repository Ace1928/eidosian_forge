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
def column_label_decoder(s: pd.Series):
    inverse_values = {value: key for key, value in self.stats_[f'unique_values({self.label_column})'].items()}
    return s.map(inverse_values)
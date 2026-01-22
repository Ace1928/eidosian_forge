from collections import Counter
from typing import Callable, List, Optional
import pandas as pd
from ray.data import Dataset
from ray.data.preprocessor import Preprocessor
from ray.data.preprocessors.utils import simple_hash, simple_split_tokenizer
from ray.util.annotations import PublicAPI
def most_common(counter: Counter, n: int):
    return Counter(dict(counter.most_common(n)))
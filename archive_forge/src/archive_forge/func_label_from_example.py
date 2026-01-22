import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import is_tf_available, logging
from .utils import DataProcessor, InputExample, InputFeatures
def label_from_example(example: InputExample) -> Union[int, float, None]:
    if example.label is None:
        return None
    if output_mode == 'classification':
        return label_map[example.label]
    elif output_mode == 'regression':
        return float(example.label)
    raise KeyError(output_mode)
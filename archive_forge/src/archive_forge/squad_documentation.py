import json
import os
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
from ...models.bert.tokenization_bert import whitespace_tokenize
from ...tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy
from ...utils import is_tf_available, is_torch_available, logging
from .utils import DataProcessor

        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        
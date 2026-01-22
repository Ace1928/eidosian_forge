import pickle
from typing import Optional, Tuple
from ...tokenization_utils_base import BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging

        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        
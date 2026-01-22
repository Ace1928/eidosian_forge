import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from .number_normalizer import EnglishNumberNormalizer
Build model inputs from a sequence by appending eos_token_id.
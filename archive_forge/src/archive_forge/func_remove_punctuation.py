import os
import re
import string
import warnings
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import AddedToken
from ...utils import logging, requires_backends
def remove_punctuation(self, text: str) -> str:
    return text.translate(str.maketrans('', '', string.punctuation))
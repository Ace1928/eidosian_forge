import os
import re
import warnings
from shutil import copyfile
from typing import List, Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
def get_sentinel_token_ids(self):
    return [self.convert_tokens_to_ids(token) for token in self.get_sentinel_tokens()]
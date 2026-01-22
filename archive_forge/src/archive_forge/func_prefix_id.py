import os
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as spm
from ...convert_slow_tokenizer import import_protobuf
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging, requires_backends
@property
def prefix_id(self):
    if self._prefix_token is None:
        return None
    return self.convert_tokens_to_ids(self.prefix_token)
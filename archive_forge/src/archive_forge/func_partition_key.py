import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
def partition_key(self) -> CharStateKey:
    return CharStateKey(token_ix=self.token_ix, anno_ix=self.anno_ix)
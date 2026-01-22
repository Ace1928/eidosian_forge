import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
@property
def is_multitoken(self):
    """
        BPE tokenizers can output more than one token for a char
        """
    return len(self.tokens) > 1
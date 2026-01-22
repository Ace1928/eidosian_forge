import json
import os
import unicodedata
from functools import lru_cache
from typing import Optional, Tuple
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
Converts a sequence of tokens (string) in a single string.
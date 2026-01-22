import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from .number_normalizer import EnglishNormalizer
Converts a sequence of tokens (string) in a single string.
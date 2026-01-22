import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging

        A simple chat template that ignores role information and just concatenates messages with EOS tokens.
        
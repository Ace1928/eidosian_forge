import collections
import json
import os
import re
from typing import Optional, Tuple
import numpy as np
from ...tokenization_utils_fast import PreTrainedTokenizer
from ...utils import logging
def check_simbol(x):
    e = x.encode()
    if len(x) == 1 and len(e) == 2:
        c = (int(e[0]) << 8) + int(e[1])
        if c >= 49825 and c <= 49855 or (c >= 51072 and c <= 51075) or (c >= 51897 and c <= 52159) or (c >= 52352 and c <= 52642):
            return True
    return False
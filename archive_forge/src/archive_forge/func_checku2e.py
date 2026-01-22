import collections
import json
import os
import re
from typing import Optional, Tuple
import numpy as np
from ...tokenization_utils_fast import PreTrainedTokenizer
from ...utils import logging
def checku2e(x):
    e = x.encode()
    if len(x) == 1 and len(e) == 3:
        c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
        if c >= 14844032 and c <= 14856319:
            return True
    return False
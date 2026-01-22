import glob
import os
import pickle
import re
from collections import Counter, OrderedDict
from typing import List, Optional, Tuple
import numpy as np
from ....tokenization_utils import PreTrainedTokenizer
from ....utils import (
def moses_tokenize(self, text):
    return self.moses_tokenizer.tokenize(text, aggressive_dash_splits=True, return_str=False, escape=False, protected_patterns=self.never_split)
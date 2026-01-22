import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
@property
def first_upper(self):
    """True if the token's first character is uppercase."""
    return self.tok[0].isupper()
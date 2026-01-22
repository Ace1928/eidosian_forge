import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
@property
def is_non_punct(self):
    """True if the token is either a number or is alphabetic."""
    return _re_non_punct.search(self.type)
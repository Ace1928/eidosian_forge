import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Match, Optional, Tuple, Union
from nltk.probability import FreqDist
from nltk.tokenize.api import TokenizerI
@property
def type_no_period(self):
    """
        The type with its final period removed if it has one.
        """
    if len(self.type) > 1 and self.type[-1] == '.':
        return self.type[:-1]
    return self.type
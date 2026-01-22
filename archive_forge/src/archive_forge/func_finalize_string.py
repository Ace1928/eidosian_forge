import re
import warnings
from enum import Enum
from math import gcd
def finalize_string(self, s):
    return insert_quotes(s, self.quotes_map)
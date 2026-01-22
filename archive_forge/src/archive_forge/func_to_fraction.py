import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
def to_fraction(s: str):
    try:
        return Fraction(s)
    except ValueError:
        return None
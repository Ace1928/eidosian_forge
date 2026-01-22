import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
def extract_cents(m: Match):
    try:
        return f'¢{int(m.group(1))}'
    except ValueError:
        return m.string
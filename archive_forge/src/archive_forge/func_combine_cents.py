import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union
import regex
def combine_cents(m: Match):
    try:
        currency = m.group(1)
        integer = m.group(2)
        cents = int(m.group(3))
        return f'{currency}{integer}.{cents:02d}'
    except ValueError:
        return m.string
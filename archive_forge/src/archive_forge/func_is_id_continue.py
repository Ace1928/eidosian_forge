import unicodedata
import os
from itertools import product
from collections import deque
from typing import Callable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, Dict, Any, Sequence, Iterable, AbstractSet
import sys, re
import logging
def is_id_continue(s: str) -> bool:
    """
    Checks if all characters in `s` are alphanumeric characters (Unicode standard, so diacritics, indian vowels, non-latin
    numbers, etc. all pass). Synonymous with a Python `ID_CONTINUE` identifier. See PEP 3131 for details.
    """
    return _test_unicode_category(s, _ID_CONTINUE)
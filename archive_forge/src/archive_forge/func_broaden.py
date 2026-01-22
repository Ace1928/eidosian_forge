from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def broaden(self) -> 'List[Language]':
    """
        Like `broader_tags`, but returrns Language objects instead of strings.
        """
    return [Language.get(tag) for tag in self.broader_tags()]
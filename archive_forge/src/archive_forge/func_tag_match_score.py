from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def tag_match_score(desired: Union[str, Language], supported: Union[str, Language]) -> int:
    """
    DEPRECATED: use .distance() instead, which uses newer data and is _lower_
    for better matching languages.

    Return a number from 0 to 100 indicating the strength of match between the
    language the user desires, D, and a supported language, S. Higher numbers
    are better. A reasonable cutoff for not messing with your users is to
    only accept scores of 75 or more.

    A score of 100 means the languages are the same, possibly after normalizing
    and filling in likely values.
    """
    warnings.warn("tag_match_score is deprecated because it's based on deprecated CLDR info. Use tag_distance instead, which is _lower_ for better matching languages. ", DeprecationWarning)
    desired_ld = Language.get(desired)
    supported_ld = Language.get(supported)
    return desired_ld.match_score(supported_ld)
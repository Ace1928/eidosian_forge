from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def closest_supported_match(desired_language: Union[str, Language], supported_languages: Sequence[str], max_distance: int=25) -> Optional[str]:
    """
    Wraps `closest_match` with a simpler return type. Returns the language
    tag of the closest match if there is one, or None if there is not.

    >>> closest_supported_match('fr', ['de', 'en', 'fr'])
    'fr'

    >>> closest_supported_match('pt', ['pt-BR', 'pt-PT'])
    'pt-BR'

    >>> closest_supported_match('en-AU', ['en-GB', 'en-US'])
    'en-GB'

    >>> closest_supported_match('und', ['en', 'und'])
    'und'

    >>> closest_supported_match('af', ['en', 'nl', 'zu'])
    'nl'

    >>> print(closest_supported_match('af', ['en', 'nl', 'zu'], max_distance=10))
    None
    """
    code, distance = closest_match(desired_language, supported_languages, max_distance)
    if distance == 1000:
        return None
    else:
        return code
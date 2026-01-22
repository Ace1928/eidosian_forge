from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def variant_names(self, language: Union[str, 'Language']=DEFAULT_LANGUAGE, max_distance: int=25) -> Sequence[str]:
    """
        Deprecated in version 3.0.

        We don't store names for variants anymore, so this just returns the list
        of variant codes, such as ['oxendict'] for en-GB-oxendict.
        """
    warnings.warn('variant_names is deprecated and just returns the variant codes', DeprecationWarning)
    return self.variants or []
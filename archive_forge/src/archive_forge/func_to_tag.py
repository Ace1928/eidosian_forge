from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def to_tag(self) -> str:
    """
        Convert a Language back to a standard language tag, as a string.
        This is also the str() representation of a Language object.

        >>> Language.make(language='en', territory='GB').to_tag()
        'en-GB'

        >>> Language.make(language='yue', script='Hant', territory='HK').to_tag()
        'yue-Hant-HK'

        >>> Language.make(script='Arab').to_tag()
        'und-Arab'

        >>> str(Language.make(territory='IN'))
        'und-IN'
        """
    if self._str_tag is not None:
        return self._str_tag
    subtags = ['und']
    if self.language:
        subtags[0] = self.language
    if self.extlangs:
        for extlang in sorted(self.extlangs):
            subtags.append(extlang)
    if self.script:
        subtags.append(self.script)
    if self.territory:
        subtags.append(self.territory)
    if self.variants:
        for variant in sorted(self.variants):
            subtags.append(variant)
    if self.extensions:
        for ext in self.extensions:
            subtags.append(ext)
    if self.private:
        subtags.append(self.private)
    self._str_tag = '-'.join(subtags)
    return self._str_tag
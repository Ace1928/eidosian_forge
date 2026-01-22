from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def broader_tags(self) -> List[str]:
    """
        Iterate through increasingly general tags for this language.

        This isn't actually that useful for matching two arbitrary language tags
        against each other, but it is useful for matching them against a known
        standardized form, such as in the CLDR data.

        The list of broader versions to try appears in UTR 35, section 4.3,
        "Likely Subtags".

        >>> Language.get('nn-Latn-NO-x-thingy').broader_tags()
        ['nn-Latn-NO-x-thingy', 'nn-Latn-NO', 'nn-NO', 'nn-Latn', 'nn', 'und-Latn', 'und']

        >>> Language.get('arb-Arab').broader_tags()
        ['arb-Arab', 'ar-Arab', 'arb', 'ar', 'und-Arab', 'und']
        """
    if self._broader is not None:
        return self._broader
    self._broader = [self.to_tag()]
    seen = set([self.to_tag()])
    for keyset in self.BROADER_KEYSETS:
        for start_language in (self, self.prefer_macrolanguage()):
            filtered = start_language._filter_attributes(keyset)
            tag = filtered.to_tag()
            if tag not in seen:
                self._broader.append(tag)
                seen.add(tag)
    return self._broader
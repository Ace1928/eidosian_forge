from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def simplify_script(self) -> 'Language':
    """
        Remove the script from some parsed language data, if the script is
        redundant with the language.

        >>> Language.make(language='en', script='Latn').simplify_script()
        Language.make(language='en')

        >>> Language.make(language='yi', script='Latn').simplify_script()
        Language.make(language='yi', script='Latn')

        >>> Language.make(language='yi', script='Hebr').simplify_script()
        Language.make(language='yi')
        """
    if self._simplified is not None:
        return self._simplified
    if self.language and self.script:
        if DEFAULT_SCRIPTS.get(self.language) == self.script:
            result = self.update_dict({'script': None})
            self._simplified = result
            return self._simplified
    self._simplified = self
    return self._simplified
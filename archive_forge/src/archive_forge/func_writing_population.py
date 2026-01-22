from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def writing_population(self) -> int:
    """
        Get an estimate of how many people in the world read and write
        this language, derived from CLDR data. Requires that `language_data`
        is installed.

        For many languages that aren't typically written, this is an
        overestimate, according to CLDR -- the data often includes people who
        speak that language but write in a different language.

        Only the language, script, and territory codes will be considered.
        If a territory code is included, the population will count only the
        speakers of the language in that territory.

        >>> all = Language.get('zh').writing_population()
        >>> all
        1240326057

        >>> traditional = Language.get('zh-Hant').writing_population()
        >>> traditional
        37019589

        >>> simplified = Language.get('zh-Hans').writing_population()
        >>> all == traditional + simplified
        True

        >>> Language.get('zh-Hant-HK').writing_population()
        6439733
        >>> Language.get('zh-Hans-HK').writing_population()
        338933

        Note that if you want to get the total Chinese writing population
        of Hong Kong, you need to avoid normalization that would interpret
        'zh-HK' as 'zh-Hant-HK'.

        >>> Language.get('zh-HK', normalize=False).writing_population()
        6778666

        Unknown or unspecified language codes get a population of 0.

        >>> Language.get('xyz').writing_population()
        0

        >>> Language.get('und').writing_population()
        0
        """
    try:
        from language_data.population_data import LANGUAGE_WRITING_POPULATION
    except ImportError:
        print(LANGUAGE_NAME_IMPORT_MESSAGE, file=sys.stdout)
        raise
    lang = self._filter_attributes(['language', 'script', 'territory'])
    if str(lang) in LANGUAGE_WRITING_POPULATION:
        return LANGUAGE_WRITING_POPULATION[str(lang)]
    else:
        lang = lang.simplify_script()
        return LANGUAGE_WRITING_POPULATION.get(str(lang), 0)
from .data_dicts import LANGUAGE_DISTANCES
from typing import Dict, Tuple

    Takes in triples of (language, script, territory), which can be derived by
    'maximizing' a language tag. Returns a number from 0 to 135 indicating the
    'distance' between these for the purposes of language matching.
    
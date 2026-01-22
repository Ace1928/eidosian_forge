import importlib
from codecs import IncrementalDecoder
from collections import Counter, OrderedDict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from .assets import FREQUENCIES
from .constant import KO_NAMES, LANGUAGE_SUPPORTED_COUNT, TOO_SMALL_SEQUENCE, ZH_NAMES
from .md import is_suspiciously_successive_range
from .models import CoherenceMatches
from .utils import (
@lru_cache(maxsize=2048)
def coherence_ratio(decoded_sequence: str, threshold: float=0.1, lg_inclusion: Optional[str]=None) -> CoherenceMatches:
    """
    Detect ANY language that can be identified in given sequence. The sequence will be analysed by layers.
    A layer = Character extraction by alphabets/ranges.
    """
    results = []
    ignore_non_latin = False
    sufficient_match_count = 0
    lg_inclusion_list = lg_inclusion.split(',') if lg_inclusion is not None else []
    if 'Latin Based' in lg_inclusion_list:
        ignore_non_latin = True
        lg_inclusion_list.remove('Latin Based')
    for layer in alpha_unicode_split(decoded_sequence):
        sequence_frequencies = Counter(layer)
        most_common = sequence_frequencies.most_common()
        character_count = sum((o for c, o in most_common))
        if character_count <= TOO_SMALL_SEQUENCE:
            continue
        popular_character_ordered = [c for c, o in most_common]
        for language in lg_inclusion_list or alphabet_languages(popular_character_ordered, ignore_non_latin):
            ratio = characters_popularity_compare(language, popular_character_ordered)
            if ratio < threshold:
                continue
            elif ratio >= 0.8:
                sufficient_match_count += 1
            results.append((language, round(ratio, 4)))
            if sufficient_match_count >= 3:
                break
    return sorted(results, key=lambda x: x[1], reverse=True)
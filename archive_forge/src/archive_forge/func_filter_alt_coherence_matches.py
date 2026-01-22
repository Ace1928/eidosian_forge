import importlib
from codecs import IncrementalDecoder
from collections import Counter
from functools import lru_cache
from typing import Counter as TypeCounter, Dict, List, Optional, Tuple
from .constant import (
from .md import is_suspiciously_successive_range
from .models import CoherenceMatches
from .utils import (
def filter_alt_coherence_matches(results: CoherenceMatches) -> CoherenceMatches:
    """
    We shall NOT return "English—" in CoherenceMatches because it is an alternative
    of "English". This function only keeps the best match and remove the em-dash in it.
    """
    index_results: Dict[str, List[float]] = dict()
    for result in results:
        language, ratio = result
        no_em_name: str = language.replace('—', '')
        if no_em_name not in index_results:
            index_results[no_em_name] = []
        index_results[no_em_name].append(ratio)
    if any((len(index_results[e]) > 1 for e in index_results)):
        filtered_results: CoherenceMatches = []
        for language in index_results:
            filtered_results.append((language, max(index_results[language])))
        return filtered_results
    return results
import sys
from typing import Dict, Any, Tuple, Callable, Iterator, List, Optional, IO
import re
from spacy import Language
from spacy.util import registry
def matcher_for_regex_patterns(regexps: Optional[List[str]]=None) -> Callable[[str], bool]:
    try:
        compiled = []
        if regexps is not None:
            for regex in regexps:
                compiled.append(re.compile(regex, flags=re.MULTILINE))
    except re.error as err:
        raise ValueError(f"Regular expression `{regex}` couldn't be compiled for logger stats matcher") from err

    def is_match(string: str) -> bool:
        for regex in compiled:
            if regex.search(string):
                return True
        return False
    return is_match
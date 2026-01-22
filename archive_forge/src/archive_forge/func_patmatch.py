import os.path
import re
from typing import Callable, Dict, Iterable, Iterator, List, Match, Optional, Pattern
from sphinx.util.osutil import canon_path, path_stabilize
def patmatch(name: str, pat: str) -> Optional[Match[str]]:
    """Return if name matches the regular expression (pattern)
    ``pat```. Adapted from fnmatch module."""
    if pat not in _pat_cache:
        _pat_cache[pat] = re.compile(_translate_pattern(pat))
    return _pat_cache[pat].match(name)
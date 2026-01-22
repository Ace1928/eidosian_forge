import os.path
import re
from typing import Callable, Dict, Iterable, Iterator, List, Match, Optional, Pattern
from sphinx.util.osutil import canon_path, path_stabilize
def compile_matchers(patterns: Iterable[str]) -> List[Callable[[str], Optional[Match[str]]]]:
    return [re.compile(_translate_pattern(pat)).match for pat in patterns]
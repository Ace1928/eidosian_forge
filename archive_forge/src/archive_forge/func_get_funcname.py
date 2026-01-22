import tokenize
from typing import Dict, List, Optional
def get_funcname(filename: str, lineno: int) -> Optional[str]:
    if filename not in cache:
        _add_file(filename)
    return cache[filename].get(lineno, None)
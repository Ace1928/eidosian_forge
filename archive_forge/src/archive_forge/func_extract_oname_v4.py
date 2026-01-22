import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def extract_oname_v4(code: str, cursor_pos: int) -> str:
    """Reimplement token-finding logic from IPython 2.x javascript

    for adapting object_info_request from v5 to v4
    """
    line, _ = code_to_line(code, cursor_pos)
    oldline = line
    line = _match_bracket.sub('', line)
    while oldline != line:
        oldline = line
        line = _match_bracket.sub('', line)
    line = _end_bracket.sub('', line)
    matches = _identifier.findall(line)
    if matches:
        return matches[-1]
    else:
        return ''
import json
import re
from typing import Any, Dict, List, Tuple
from ._version import protocol_version_info
def inspect_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Handle an inspect request."""
    content = msg['content']
    name = content['oname']
    new_content = msg['content'] = {}
    new_content['code'] = name
    new_content['cursor_pos'] = len(name)
    new_content['detail_level'] = content['detail_level']
    return msg
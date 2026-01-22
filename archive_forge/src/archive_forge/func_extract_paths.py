import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
def extract_paths(self, query: str) -> 'List[str]':
    """
        Args:
            query: cypher query
        """
    paths = []
    idx = 0
    while (matched := self.path_pattern.findall(query[idx:])):
        matched = matched[0]
        matched = [m for i, m in enumerate(matched) if i not in [1, len(matched) - 1]]
        path = ''.join(matched)
        idx = query.find(path) + len(path) - len(matched[-1])
        paths.append(path)
    return paths
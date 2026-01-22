import re
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple
def detect_node_variables(self, query: str) -> Dict[str, List[str]]:
    """
        Args:
            query: cypher query
        """
    nodes = re.findall(self.node_pattern, query)
    nodes = [self.clean_node(node) for node in nodes]
    res: Dict[str, Any] = {}
    for node in nodes:
        parts = node.split(':')
        if parts == '':
            continue
        variable = parts[0]
        if variable not in res:
            res[variable] = []
        res[variable] += parts[1:]
    return res
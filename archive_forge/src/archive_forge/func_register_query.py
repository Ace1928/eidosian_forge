from typing import Any, Dict, List, Optional
from langchain_community.graphs.graph_store import GraphStore
def register_query(self, function_header: str, description: str, docstring: str, param_types: dict={}) -> List[str]:
    """
        Wrapper function to register a custom GSQL query to the TigerGraph NLQS.
        """
    return self._conn.ai.registerCustomQuery(function_header, description, docstring, param_types)
from .ast import (
def is_definition_node(node: Node) -> bool:
    """Check whether the given node represents a definition."""
    return isinstance(node, DefinitionNode)
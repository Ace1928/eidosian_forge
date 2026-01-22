import typing as t
from .nodes import Node
def visit_list(self, node: Node, *args: t.Any, **kwargs: t.Any) -> t.List[Node]:
    """As transformers may return lists in some places this method
        can be used to enforce a list as return value.
        """
    rv = self.visit(node, *args, **kwargs)
    if not isinstance(rv, list):
        return [rv]
    return rv
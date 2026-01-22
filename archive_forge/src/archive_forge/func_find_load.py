import typing as t
from . import nodes
from .visitor import NodeVisitor
def find_load(self, target: str) -> t.Optional[t.Any]:
    if target in self.loads:
        return self.loads[target]
    if self.parent is not None:
        return self.parent.find_load(target)
    return None
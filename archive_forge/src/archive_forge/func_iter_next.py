import weakref
from weakref import ReferenceType
def iter_next(self, *, skip_current=False):
    node = self.next_node if skip_current else self
    while node:
        yield node
        node = node.next_node
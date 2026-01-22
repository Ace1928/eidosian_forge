import random
import sys
from . import Nodes
def set_subtree(self, node):
    """Return subtree as a set of nested sets.

        sets = set_subtree(self,node)
        """
    if self.node(node).succ == []:
        return self.node(node).data.taxon
    else:
        try:
            return frozenset((self.set_subtree(n) for n in self.node(node).succ))
        except Exception:
            print(node)
            print(self.node(node).succ)
            for n in self.node(node).succ:
                print(f'{n} {self.set_subtree(n)}')
            print([self.set_subtree(n) for n in self.node(node).succ])
            raise
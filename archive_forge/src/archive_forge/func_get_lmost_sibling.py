import numpy as np
def get_lmost_sibling(self):
    if not self._lmost_sibling and self.parent and (self != self.parent.children[0]):
        self._lmost_sibling = self.parent.children[0]
    return self._lmost_sibling
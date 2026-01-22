import numpy as np
def max_extents(self):
    extents = [c.max_extents() for c in self.children]
    extents.append((self.x, self.y))
    return np.max(extents, axis=0)
import networkx as nx
def width(self, L):
    m = 0
    for i, row in enumerate(L):
        w = 0
        x, y = np.nonzero(row)
        if len(y) > 0:
            v = y - i
            w = v.max() - v.min() + 1
            m = max(w, m)
    return m
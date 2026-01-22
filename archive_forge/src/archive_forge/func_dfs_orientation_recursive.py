from collections import defaultdict
import networkx as nx
def dfs_orientation_recursive(self, v):
    """Recursive version of :meth:`dfs_orientation`."""
    e = self.parent_edge[v]
    for w in self.G[v]:
        if (v, w) in self.DG.edges or (w, v) in self.DG.edges:
            continue
        vw = (v, w)
        self.DG.add_edge(v, w)
        self.lowpt[vw] = self.height[v]
        self.lowpt2[vw] = self.height[v]
        if self.height[w] is None:
            self.parent_edge[w] = vw
            self.height[w] = self.height[v] + 1
            self.dfs_orientation_recursive(w)
        else:
            self.lowpt[vw] = self.height[w]
        self.nesting_depth[vw] = 2 * self.lowpt[vw]
        if self.lowpt2[vw] < self.height[v]:
            self.nesting_depth[vw] += 1
        if e is not None:
            if self.lowpt[vw] < self.lowpt[e]:
                self.lowpt2[e] = min(self.lowpt[e], self.lowpt2[vw])
                self.lowpt[e] = self.lowpt[vw]
            elif self.lowpt[vw] > self.lowpt[e]:
                self.lowpt2[e] = min(self.lowpt2[e], self.lowpt[vw])
            else:
                self.lowpt2[e] = min(self.lowpt2[e], self.lowpt2[vw])
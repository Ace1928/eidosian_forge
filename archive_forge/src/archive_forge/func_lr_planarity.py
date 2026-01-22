from collections import defaultdict
import networkx as nx
def lr_planarity(self):
    """Execute the LR planarity test.

        Returns
        -------
        embedding : dict
            If the graph is planar an embedding is returned. Otherwise None.
        """
    if self.G.order() > 2 and self.G.size() > 3 * self.G.order() - 6:
        return None
    for v in self.G:
        self.adjs[v] = list(self.G[v])
    for v in self.G:
        if self.height[v] is None:
            self.height[v] = 0
            self.roots.append(v)
            self.dfs_orientation(v)
    self.G = None
    self.lowpt2 = None
    self.adjs = None
    for v in self.DG:
        self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
    for v in self.roots:
        if not self.dfs_testing(v):
            return None
    self.height = None
    self.lowpt = None
    self.S = None
    self.stack_bottom = None
    self.lowpt_edge = None
    for e in self.DG.edges:
        self.nesting_depth[e] = self.sign(e) * self.nesting_depth[e]
    self.embedding.add_nodes_from(self.DG.nodes)
    for v in self.DG:
        self.ordered_adjs[v] = sorted(self.DG[v], key=lambda x: self.nesting_depth[v, x])
        previous_node = None
        for w in self.ordered_adjs[v]:
            self.embedding.add_half_edge_cw(v, w, previous_node)
            previous_node = w
    self.DG = None
    self.nesting_depth = None
    self.ref = None
    for v in self.roots:
        self.dfs_embedding(v)
    self.roots = None
    self.parent_edge = None
    self.ordered_adjs = None
    self.left_ref = None
    self.right_ref = None
    self.side = None
    return self.embedding
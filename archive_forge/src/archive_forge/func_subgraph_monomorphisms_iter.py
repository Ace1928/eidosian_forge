import sys
def subgraph_monomorphisms_iter(self):
    """Generator over monomorphisms between a subgraph of G1 and G2."""
    self.test = 'mono'
    self.initialize()
    yield from self.match()
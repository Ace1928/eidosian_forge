import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_faster_could_be_isomorphic(self):
    assert iso.faster_could_be_isomorphic(self.G3, self.G2)
    assert not iso.faster_could_be_isomorphic(self.G3, self.G5)
    assert not iso.faster_could_be_isomorphic(self.G1, self.G6)
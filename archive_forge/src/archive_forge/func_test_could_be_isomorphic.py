import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_could_be_isomorphic(self):
    assert iso.could_be_isomorphic(self.G1, self.G2)
    assert iso.could_be_isomorphic(self.G1, self.G3)
    assert not iso.could_be_isomorphic(self.G1, self.G4)
    assert iso.could_be_isomorphic(self.G3, self.G2)
    assert not iso.could_be_isomorphic(self.G1, self.G6)
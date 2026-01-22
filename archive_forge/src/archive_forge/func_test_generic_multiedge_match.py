from operator import eq
import networkx as nx
from networkx.algorithms import isomorphism as iso
def test_generic_multiedge_match(self):
    full_match = iso.generic_multiedge_match(['id', 'flowMin', 'flowMax'], [None] * 3, [eq] * 3)
    flow_match = iso.generic_multiedge_match(['flowMin', 'flowMax'], [None] * 2, [eq] * 2)
    min_flow_match = iso.generic_multiedge_match('flowMin', None, eq)
    id_match = iso.generic_multiedge_match('id', None, eq)
    assert flow_match(self.G1[1][2], self.G2[2][3])
    assert min_flow_match(self.G1[1][2], self.G2[2][3])
    assert id_match(self.G1[1][2], self.G2[2][3])
    assert full_match(self.G1[1][2], self.G2[2][3])
    assert flow_match(self.G3[3][4], self.G4[4][5])
    assert min_flow_match(self.G3[3][4], self.G4[4][5])
    assert not id_match(self.G3[3][4], self.G4[4][5])
    assert not full_match(self.G3[3][4], self.G4[4][5])
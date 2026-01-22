import pytest
import networkx as nx
def test_kamada_kawai_costfn_1d(self):
    costfn = nx.drawing.layout._kamada_kawai_costfn
    pos = np.array([4.0, 7.0])
    invdist = 1 / np.array([[0.1, 2.0], [2.0, 0.3]])
    cost, grad = costfn(pos, np, invdist, meanweight=0, dim=1)
    assert cost == pytest.approx((3 / 2.0 - 1) ** 2, abs=1e-07)
    assert grad[0] == pytest.approx(-0.5, abs=1e-07)
    assert grad[1] == pytest.approx(0.5, abs=1e-07)
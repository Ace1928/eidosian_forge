import pytest
import networkx as nx
def check_kamada_kawai_costfn(self, pos, invdist, meanwt, dim):
    costfn = nx.drawing.layout._kamada_kawai_costfn
    cost, grad = costfn(pos.ravel(), np, invdist, meanweight=meanwt, dim=dim)
    expected_cost = 0.5 * meanwt * np.sum(np.sum(pos, axis=0) ** 2)
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            diff = np.linalg.norm(pos[i] - pos[j])
            expected_cost += (diff * invdist[i][j] - 1.0) ** 2
    assert cost == pytest.approx(expected_cost, abs=1e-07)
    dx = 0.0001
    for nd in range(pos.shape[0]):
        for dm in range(pos.shape[1]):
            idx = nd * pos.shape[1] + dm
            ps = pos.flatten()
            ps[idx] += dx
            cplus = costfn(ps, np, invdist, meanweight=meanwt, dim=pos.shape[1])[0]
            ps[idx] -= 2 * dx
            cminus = costfn(ps, np, invdist, meanweight=meanwt, dim=pos.shape[1])[0]
            assert grad[idx] == pytest.approx((cplus - cminus) / (2 * dx), abs=1e-05)
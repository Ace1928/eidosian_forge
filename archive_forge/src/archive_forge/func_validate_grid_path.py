import pytest
import networkx as nx
def validate_grid_path(r, c, s, t, p):
    assert isinstance(p, list)
    assert p[0] == s
    assert p[-1] == t
    s = ((s - 1) // c, (s - 1) % c)
    t = ((t - 1) // c, (t - 1) % c)
    assert len(p) == abs(t[0] - s[0]) + abs(t[1] - s[1]) + 1
    p = [((u - 1) // c, (u - 1) % c) for u in p]
    for u in p:
        assert 0 <= u[0] < r
        assert 0 <= u[1] < c
    for u, v in zip(p[:-1], p[1:]):
        assert (abs(v[0] - u[0]), abs(v[1] - u[1])) in [(0, 1), (1, 0)]
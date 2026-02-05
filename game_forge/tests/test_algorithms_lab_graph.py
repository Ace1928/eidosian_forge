import numpy as np

from algorithms_lab.core import Domain, WrapMode
from algorithms_lab.graph import build_neighbor_graph


def test_build_neighbor_graph_wrap():
    positions = np.array(
        [
            [0.05, 0.05],
            [0.2, 0.05],
            [0.95, 0.05],
        ],
        dtype=np.float32,
    )
    domain = Domain(
        mins=np.array([0.0, 0.0], dtype=np.float32),
        maxs=np.array([1.0, 1.0], dtype=np.float32),
        wrap=WrapMode.WRAP,
    )
    graph = build_neighbor_graph(positions, radius=0.3, domain=domain, method="grid", backend="numpy")
    edges = set(zip(graph.rows.tolist(), graph.cols.tolist()))
    expected_edges = {
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 2),
        (2, 1),
    }
    assert expected_edges.issubset(edges)
    idx = np.where((graph.rows == 0) & (graph.cols == 2))[0]
    assert idx.size == 1
    assert np.isclose(graph.dist[idx[0]], 0.1, atol=1e-3)


def test_build_neighbor_graph_3d():
    positions = np.array(
        [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    domain = Domain(
        mins=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        maxs=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        wrap=WrapMode.NONE,
    )
    graph = build_neighbor_graph(positions, radius=0.25, domain=domain, method="grid", backend="numpy")
    edges = set(zip(graph.rows.tolist(), graph.cols.tolist()))
    assert edges == {(0, 1), (1, 0)}
    assert np.isclose(graph.dist[0], 0.1, atol=1e-4)

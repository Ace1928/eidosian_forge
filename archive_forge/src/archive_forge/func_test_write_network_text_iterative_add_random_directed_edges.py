import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_iterative_add_random_directed_edges():
    """
    Walk through the cases going from a disconnected to fully connected graph
    """
    rng = random.Random(724466096)
    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2, 3, 4, 5])
    possible_edges = list(product(graph.nodes, graph.nodes))
    rng.shuffle(possible_edges)
    graph.add_edges_from(possible_edges[0:8])
    lines = []
    write = lines.append
    write('--- initial state ---')
    nx.write_network_text(graph, path=write, end='')
    for i, j in possible_edges[8:12]:
        write(f'--- add_edge({i}, {j}) ---')
        graph.add_edge(i, j)
        nx.write_network_text(graph, path=write, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- initial state ---\n        ╙── 3 ╾ 5\n            └─╼ 2 ╾ 2\n                ├─╼ 4 ╾ 4\n                │   ├─╼ 5\n                │   │   ├─╼ 1 ╾ 1\n                │   │   │   └─╼  ...\n                │   │   └─╼  ...\n                │   └─╼  ...\n                └─╼  ...\n        --- add_edge(4, 1) ---\n        ╙── 3 ╾ 5\n            └─╼ 2 ╾ 2\n                ├─╼ 4 ╾ 4\n                │   ├─╼ 5\n                │   │   ├─╼ 1 ╾ 1, 4\n                │   │   │   └─╼  ...\n                │   │   └─╼  ...\n                │   └─╼  ...\n                └─╼  ...\n        --- add_edge(2, 1) ---\n        ╙── 3 ╾ 5\n            └─╼ 2 ╾ 2\n                ├─╼ 4 ╾ 4\n                │   ├─╼ 5\n                │   │   ├─╼ 1 ╾ 1, 4, 2\n                │   │   │   └─╼  ...\n                │   │   └─╼  ...\n                │   └─╼  ...\n                └─╼  ...\n        --- add_edge(5, 2) ---\n        ╙── 3 ╾ 5\n            └─╼ 2 ╾ 2, 5\n                ├─╼ 4 ╾ 4\n                │   ├─╼ 5\n                │   │   ├─╼ 1 ╾ 1, 4, 2\n                │   │   │   └─╼  ...\n                │   │   └─╼  ...\n                │   └─╼  ...\n                └─╼  ...\n        --- add_edge(1, 5) ---\n        ╙── 3 ╾ 5\n            └─╼ 2 ╾ 2, 5\n                ├─╼ 4 ╾ 4\n                │   ├─╼ 5 ╾ 1\n                │   │   ├─╼ 1 ╾ 1, 4, 2\n                │   │   │   └─╼  ...\n                │   │   └─╼  ...\n                │   └─╼  ...\n                └─╼  ...\n\n        ').strip()
    assert target == text
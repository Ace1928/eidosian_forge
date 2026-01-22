import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_iterative_add_directed_edges():
    """
    Walk through the cases going from a disconnected to fully connected graph
    """
    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2, 3, 4])
    lines = []
    write = lines.append
    write('--- initial state ---')
    nx.write_network_text(graph, path=write, end='')
    for i, j in product(graph.nodes, graph.nodes):
        write(f'--- add_edge({i}, {j}) ---')
        graph.add_edge(i, j)
        nx.write_network_text(graph, path=write, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- initial state ---\n        ╟── 1\n        ╟── 2\n        ╟── 3\n        ╙── 4\n        --- add_edge(1, 1) ---\n        ╟── 1 ╾ 1\n        ╎   └─╼  ...\n        ╟── 2\n        ╟── 3\n        ╙── 4\n        --- add_edge(1, 2) ---\n        ╟── 1 ╾ 1\n        ╎   ├─╼ 2\n        ╎   └─╼  ...\n        ╟── 3\n        ╙── 4\n        --- add_edge(1, 3) ---\n        ╟── 1 ╾ 1\n        ╎   ├─╼ 2\n        ╎   ├─╼ 3\n        ╎   └─╼  ...\n        ╙── 4\n        --- add_edge(1, 4) ---\n        ╙── 1 ╾ 1\n            ├─╼ 2\n            ├─╼ 3\n            ├─╼ 4\n            └─╼  ...\n        --- add_edge(2, 1) ---\n        ╙── 2 ╾ 1\n            └─╼ 1 ╾ 1\n                ├─╼ 3\n                ├─╼ 4\n                └─╼  ...\n        --- add_edge(2, 2) ---\n        ╙── 1 ╾ 1, 2\n            ├─╼ 2 ╾ 2\n            │   └─╼  ...\n            ├─╼ 3\n            ├─╼ 4\n            └─╼  ...\n        --- add_edge(2, 3) ---\n        ╙── 1 ╾ 1, 2\n            ├─╼ 2 ╾ 2\n            │   ├─╼ 3 ╾ 1\n            │   └─╼  ...\n            ├─╼ 4\n            └─╼  ...\n        --- add_edge(2, 4) ---\n        ╙── 1 ╾ 1, 2\n            ├─╼ 2 ╾ 2\n            │   ├─╼ 3 ╾ 1\n            │   ├─╼ 4 ╾ 1\n            │   └─╼  ...\n            └─╼  ...\n        --- add_edge(3, 1) ---\n        ╙── 2 ╾ 1, 2\n            ├─╼ 1 ╾ 1, 3\n            │   ├─╼ 3 ╾ 2\n            │   │   └─╼  ...\n            │   ├─╼ 4 ╾ 2\n            │   └─╼  ...\n            └─╼  ...\n        --- add_edge(3, 2) ---\n        ╙── 3 ╾ 1, 2\n            ├─╼ 1 ╾ 1, 2\n            │   ├─╼ 2 ╾ 2, 3\n            │   │   ├─╼ 4 ╾ 1\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        --- add_edge(3, 3) ---\n        ╙── 1 ╾ 1, 2, 3\n            ├─╼ 2 ╾ 2, 3\n            │   ├─╼ 3 ╾ 1, 3\n            │   │   └─╼  ...\n            │   ├─╼ 4 ╾ 1\n            │   └─╼  ...\n            └─╼  ...\n        --- add_edge(3, 4) ---\n        ╙── 1 ╾ 1, 2, 3\n            ├─╼ 2 ╾ 2, 3\n            │   ├─╼ 3 ╾ 1, 3\n            │   │   ├─╼ 4 ╾ 1, 2\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        --- add_edge(4, 1) ---\n        ╙── 2 ╾ 1, 2, 3\n            ├─╼ 1 ╾ 1, 3, 4\n            │   ├─╼ 3 ╾ 2, 3\n            │   │   ├─╼ 4 ╾ 1, 2\n            │   │   │   └─╼  ...\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        --- add_edge(4, 2) ---\n        ╙── 3 ╾ 1, 2, 3\n            ├─╼ 1 ╾ 1, 2, 4\n            │   ├─╼ 2 ╾ 2, 3, 4\n            │   │   ├─╼ 4 ╾ 1, 3\n            │   │   │   └─╼  ...\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        --- add_edge(4, 3) ---\n        ╙── 4 ╾ 1, 2, 3\n            ├─╼ 1 ╾ 1, 2, 3\n            │   ├─╼ 2 ╾ 2, 3, 4\n            │   │   ├─╼ 3 ╾ 1, 3, 4\n            │   │   │   └─╼  ...\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        --- add_edge(4, 4) ---\n        ╙── 1 ╾ 1, 2, 3, 4\n            ├─╼ 2 ╾ 2, 3, 4\n            │   ├─╼ 3 ╾ 1, 3, 4\n            │   │   ├─╼ 4 ╾ 1, 2, 4\n            │   │   │   └─╼  ...\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        ').strip()
    assert target == text
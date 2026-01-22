import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_collapse_undirected():
    graph = nx.balanced_tree(r=2, h=3, create_using=nx.Graph)
    lines = []
    write = lines.append
    write('--- Original ---')
    nx.write_network_text(graph, path=write, end='', sources=[0])
    graph.nodes[1]['collapse'] = True
    write('--- Collapse Node 1 ---')
    nx.write_network_text(graph, path=write, end='', sources=[0])
    write('--- Add alternate path (5, 3) to collapsed zone')
    graph.add_edge(5, 3)
    nx.write_network_text(graph, path=write, end='', sources=[0])
    write('--- Collapse Node 0 ---')
    graph.nodes[0]['collapse'] = True
    nx.write_network_text(graph, path=write, end='', sources=[0])
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- Original ---\n        ╙── 0\n            ├── 1\n            │   ├── 3\n            │   │   ├── 7\n            │   │   └── 8\n            │   └── 4\n            │       ├── 9\n            │       └── 10\n            └── 2\n                ├── 5\n                │   ├── 11\n                │   └── 12\n                └── 6\n                    ├── 13\n                    └── 14\n        --- Collapse Node 1 ---\n        ╙── 0\n            ├── 1 ─ 3, 4\n            │   └──  ...\n            └── 2\n                ├── 5\n                │   ├── 11\n                │   └── 12\n                └── 6\n                    ├── 13\n                    └── 14\n        --- Add alternate path (5, 3) to collapsed zone\n        ╙── 0\n            ├── 1 ─ 3, 4\n            │   └──  ...\n            └── 2\n                ├── 5\n                │   ├── 11\n                │   ├── 12\n                │   └── 3 ─ 1\n                │       ├── 7\n                │       └── 8\n                └── 6\n                    ├── 13\n                    └── 14\n        --- Collapse Node 0 ---\n        ╙── 0 ─ 1, 2\n            └──  ...\n        ').strip()
    assert target == text
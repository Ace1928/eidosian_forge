import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_circular_ladder_graph():
    graph = nx.circular_ladder_graph(4, create_using=nx.Graph)
    lines = []
    write = lines.append
    nx.write_network_text(graph, path=write, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        ╙── 0\n            ├── 1\n            │   ├── 2\n            │   │   ├── 3 ─ 0\n            │   │   │   └── 7\n            │   │   │       ├── 6 ─ 2\n            │   │   │       │   └── 5 ─ 1\n            │   │   │       │       └── 4 ─ 0, 7\n            │   │   │       └──  ...\n            │   │   └──  ...\n            │   └──  ...\n            └──  ...\n        ').strip()
    assert target == text
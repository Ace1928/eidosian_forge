import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_wheel_graph():
    graph = nx.wheel_graph(7, create_using=nx.Graph)
    lines = []
    write = lines.append
    nx.write_network_text(graph, path=write, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        ╙── 1\n            ├── 0\n            │   ├── 2 ─ 1\n            │   │   └── 3 ─ 0\n            │   │       └── 4 ─ 0\n            │   │           └── 5 ─ 0\n            │   │               └── 6 ─ 0, 1\n            │   └──  ...\n            └──  ...\n        ').strip()
    assert target == text
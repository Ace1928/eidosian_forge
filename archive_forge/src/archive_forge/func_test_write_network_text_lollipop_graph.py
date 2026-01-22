import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_lollipop_graph():
    graph = nx.lollipop_graph(4, 2, create_using=nx.Graph)
    lines = []
    write = lines.append
    nx.write_network_text(graph, path=write, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        ╙── 5\n            └── 4\n                └── 3\n                    ├── 0\n                    │   ├── 1 ─ 3\n                    │   │   └── 2 ─ 0, 3\n                    │   └──  ...\n                    └──  ...\n        ').strip()
    assert target == text
import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_multiple_sources():
    g = nx.DiGraph()
    g.add_edge(1, 2)
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 5)
    g.add_edge(3, 6)
    g.add_edge(5, 4)
    g.add_edge(4, 1)
    g.add_edge(1, 5)
    lines = []
    write = lines.append
    nodes = sorted(g.nodes())
    for n in nodes:
        write(f'--- source node: {n} ---')
        nx.write_network_text(g, path=write, sources=[n], end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- source node: 1 ---\n        ╙── 1 ╾ 4\n            ├─╼ 2\n            │   └─╼ 4 ╾ 5\n            │       └─╼  ...\n            ├─╼ 3\n            │   ├─╼ 5 ╾ 1\n            │   │   └─╼  ...\n            │   └─╼ 6\n            └─╼  ...\n        --- source node: 2 ---\n        ╙── 2 ╾ 1\n            └─╼ 4 ╾ 5\n                └─╼ 1\n                    ├─╼ 3\n                    │   ├─╼ 5 ╾ 1\n                    │   │   └─╼  ...\n                    │   └─╼ 6\n                    └─╼  ...\n        --- source node: 3 ---\n        ╙── 3 ╾ 1\n            ├─╼ 5 ╾ 1\n            │   └─╼ 4 ╾ 2\n            │       └─╼ 1\n            │           ├─╼ 2\n            │           │   └─╼  ...\n            │           └─╼  ...\n            └─╼ 6\n        --- source node: 4 ---\n        ╙── 4 ╾ 2, 5\n            └─╼ 1\n                ├─╼ 2\n                │   └─╼  ...\n                ├─╼ 3\n                │   ├─╼ 5 ╾ 1\n                │   │   └─╼  ...\n                │   └─╼ 6\n                └─╼  ...\n        --- source node: 5 ---\n        ╙── 5 ╾ 3, 1\n            └─╼ 4 ╾ 2\n                └─╼ 1\n                    ├─╼ 2\n                    │   └─╼  ...\n                    ├─╼ 3\n                    │   ├─╼ 6\n                    │   └─╼  ...\n                    └─╼  ...\n        --- source node: 6 ---\n        ╙── 6 ╾ 3\n        ').strip()
    assert target == text
import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_forest_str_overspecified_sources():
    """
    When sources are directly specified, we won't be able to determine when we
    are in the last component, so there will always be a trailing, leftmost
    pipe.
    """
    graph = nx.disjoint_union_all([nx.balanced_tree(r=2, h=1, create_using=nx.DiGraph), nx.balanced_tree(r=1, h=2, create_using=nx.DiGraph), nx.balanced_tree(r=2, h=1, create_using=nx.DiGraph)])
    target1 = dedent('\n        ╟── 0\n        ╎   ├─╼ 1\n        ╎   └─╼ 2\n        ╟── 3\n        ╎   └─╼ 4\n        ╎       └─╼ 5\n        ╟── 6\n        ╎   ├─╼ 7\n        ╎   └─╼ 8\n        ').strip()
    target2 = dedent('\n        ╟── 0\n        ╎   ├─╼ 1\n        ╎   └─╼ 2\n        ╟── 3\n        ╎   └─╼ 4\n        ╎       └─╼ 5\n        ╙── 6\n            ├─╼ 7\n            └─╼ 8\n        ').strip()
    lines = []
    nx.forest_str(graph, write=lines.append, sources=graph.nodes)
    got1 = '\n'.join(lines)
    print('got1: ')
    print(got1)
    lines = []
    nx.forest_str(graph, write=lines.append)
    got2 = '\n'.join(lines)
    print('got2: ')
    print(got2)
    assert got1 == target1
    assert got2 == target2
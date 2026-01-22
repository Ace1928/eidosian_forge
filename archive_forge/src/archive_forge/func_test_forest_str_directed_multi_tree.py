import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_forest_str_directed_multi_tree():
    tree1 = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    tree2 = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    forest = nx.disjoint_union_all([tree1, tree2])
    ret = nx.forest_str(forest)
    print(ret)
    target = dedent('\n        ╟── 0\n        ╎   ├─╼ 1\n        ╎   │   ├─╼ 3\n        ╎   │   └─╼ 4\n        ╎   └─╼ 2\n        ╎       ├─╼ 5\n        ╎       └─╼ 6\n        ╙── 7\n            ├─╼ 8\n            │   ├─╼ 10\n            │   └─╼ 11\n            └─╼ 9\n                ├─╼ 12\n                └─╼ 13\n        ').strip()
    assert ret == target
    tree3 = nx.balanced_tree(r=2, h=2, create_using=nx.DiGraph)
    forest = nx.disjoint_union_all([tree1, tree2, tree3])
    ret = nx.forest_str(forest, sources=[0, 14, 7])
    print(ret)
    target = dedent('\n        ╟── 0\n        ╎   ├─╼ 1\n        ╎   │   ├─╼ 3\n        ╎   │   └─╼ 4\n        ╎   └─╼ 2\n        ╎       ├─╼ 5\n        ╎       └─╼ 6\n        ╟── 14\n        ╎   ├─╼ 15\n        ╎   │   ├─╼ 17\n        ╎   │   └─╼ 18\n        ╎   └─╼ 16\n        ╎       ├─╼ 19\n        ╎       └─╼ 20\n        ╙── 7\n            ├─╼ 8\n            │   ├─╼ 10\n            │   └─╼ 11\n            └─╼ 9\n                ├─╼ 12\n                └─╼ 13\n        ').strip()
    assert ret == target
    ret = nx.forest_str(forest, sources=[0, 14, 7], ascii_only=True)
    print(ret)
    target = dedent('\n        +-- 0\n        :   |-> 1\n        :   |   |-> 3\n        :   |   L-> 4\n        :   L-> 2\n        :       |-> 5\n        :       L-> 6\n        +-- 14\n        :   |-> 15\n        :   |   |-> 17\n        :   |   L-> 18\n        :   L-> 16\n        :       |-> 19\n        :       L-> 20\n        +-- 7\n            |-> 8\n            |   |-> 10\n            |   L-> 11\n            L-> 9\n                |-> 12\n                L-> 13\n        ').strip()
    assert ret == target
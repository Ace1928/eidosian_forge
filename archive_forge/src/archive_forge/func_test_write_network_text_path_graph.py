import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_path_graph():
    graph = nx.path_graph(3, create_using=nx.Graph)
    lines = []
    write = lines.append
    nx.write_network_text(graph, path=write, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        ╙── 0\n            └── 1\n                └── 2\n        ').strip()
    assert target == text
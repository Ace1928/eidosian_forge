import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
def parse_graph():
    curr_token, dct = parse_kv(next(tokens))
    if curr_token.category is not None:
        unexpected(curr_token, 'EOF')
    if 'graph' not in dct:
        raise NetworkXError('input contains no graph')
    graph = dct['graph']
    if isinstance(graph, list):
        raise NetworkXError('input contains more than one graph')
    return graph
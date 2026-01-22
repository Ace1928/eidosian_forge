import time
from collections import defaultdict
from functools import partial
from typing import DefaultDict
import torch
def inline_graph(subgraph, name, node):
    rec_value_map = {inp.unique(): value_map[val.unique()] for inp, val in zip(subgraph.inputs(), node.inputs())}
    visualize_rec(graph=subgraph, value_map=rec_value_map, name_prefix=name, pb_graph=pb_graph)
    for out, val in zip(subgraph.outputs(), node.outputs()):
        value_map[val.unique()] = rec_value_map[out.unique()]
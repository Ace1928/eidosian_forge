import itertools
import numpy as np
import pandas as pd
import param
from ..core import Dataset
from ..core.boundingregion import BoundingBox
from ..core.data import PandasInterface, default_datatype
from ..core.operation import Operation
from ..core.sheetcoords import Slice
from ..core.util import (
def connect_edges_pd(graph):
    """
    Given a Graph element containing abstract edges compute edge
    segments directly connecting the source and target nodes. This
    operation depends on pandas and is a lot faster than the pure
    NumPy equivalent.
    """
    edges = graph.dframe()
    edges.index.name = 'graph_edge_index'
    edges = edges.reset_index()
    nodes = graph.nodes.dframe()
    src, tgt = graph.kdims
    x, y, idx = graph.nodes.kdims[:3]
    df = pd.merge(edges, nodes, left_on=[src.name], right_on=[idx.name])
    df = df.rename(columns={x.name: 'src_x', y.name: 'src_y'})
    df = pd.merge(df, nodes, left_on=[tgt.name], right_on=[idx.name])
    df = df.rename(columns={x.name: 'dst_x', y.name: 'dst_y'})
    df = df.sort_values('graph_edge_index').drop(['graph_edge_index'], axis=1)
    cols = ['src_x', 'src_y', 'dst_x', 'dst_y']
    edge_segments = list(df[cols].values.reshape(df.index.size, 2, 2))
    return edge_segments
import copy
import logging
import os
import re
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from typing import Set, Dict, Tuple, List
def protos_to_graph_def(net_defs, shapes=None, **kwargs):
    """
    Convert a set of Caffe2 net definitions to a Tensorflow graph.

    Args:
        net_defs: List of caffe2_pb2.NetDef protobufs representing computation
            graphs.
        shapes: Dictionary mapping blob names to their shapes/dimensions.

    Returns:
        Call to _operators_to_graph_def() with the extracted operators from the
            NetDefs and **kwargs. See _operators_to_graph_def for detailed
            **kwargs.
    """
    for net in net_defs:
        _propagate_device_option(net)
    shapes = copy.deepcopy(shapes or {})
    ops = [op for net_def in net_defs for op in net_def.op]
    return _operators_to_graph_def(shapes, ops, **kwargs)
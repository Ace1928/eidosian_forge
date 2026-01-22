from typing import Optional
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
def attr_value_proto(dtype, shape, s):
    """Create a dict of objects matching a NodeDef's attr field.

    Follows https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/attr_value.proto
    specifically designed for a NodeDef. The values have been reverse engineered from
    standard TensorBoard logged data.
    """
    attr = {}
    if s is not None:
        attr['attr'] = AttrValue(s=s.encode(encoding='utf_8'))
    if shape is not None:
        shapeproto = tensor_shape_proto(shape)
        attr['_output_shapes'] = AttrValue(list=AttrValue.ListValue(shape=[shapeproto]))
    return attr
import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def get_node_attr_value(node: NodeProto, attr_name: str) -> Any:
    matching = [x for x in node.attribute if x.name == attr_name]
    if len(matching) > 1:
        raise ValueError(f'Node has multiple attributes with name {attr_name}')
    if len(matching) < 1:
        raise ValueError(f'Node has no attribute with name {attr_name}')
    return get_attribute_value(matching[0])
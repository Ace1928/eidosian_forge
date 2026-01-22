import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def make_value_info(name: str, type_proto: TypeProto, doc_string: str='') -> ValueInfoProto:
    """Makes a ValueInfoProto with the given type_proto."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string
    value_info_proto.type.CopyFrom(type_proto)
    return value_info_proto
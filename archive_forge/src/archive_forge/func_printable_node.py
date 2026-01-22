import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def printable_node(node: NodeProto, prefix: str='', subgraphs: bool=False) -> Union[str, Tuple[str, List[GraphProto]]]:
    content = []
    if len(node.output):
        content.append(', '.join([f'%{name}' for name in node.output]))
        content.append('=')
    graphs: List[GraphProto] = []
    printed_attrs = []
    for attr in node.attribute:
        if subgraphs:
            printed_attr_subgraphs = printable_attribute(attr, subgraphs)
            if not isinstance(printed_attr_subgraphs[1], list):
                raise TypeError(f'printed_attr_subgraphs[1] must be an instance of {list}.')
            graphs.extend(printed_attr_subgraphs[1])
            printed_attrs.append(printed_attr_subgraphs[0])
        else:
            printed = printable_attribute(attr)
            if not isinstance(printed, str):
                raise TypeError(f'printed must be an instance of {str}.')
            printed_attrs.append(printed)
    printed_attributes = ', '.join(sorted(printed_attrs))
    printed_inputs = ', '.join([f'%{name}' for name in node.input])
    if node.attribute:
        content.append(f'{node.op_type}[{printed_attributes}]({printed_inputs})')
    else:
        content.append(f'{node.op_type}({printed_inputs})')
    if subgraphs:
        return (prefix + ' '.join(content), graphs)
    return prefix + ' '.join(content)
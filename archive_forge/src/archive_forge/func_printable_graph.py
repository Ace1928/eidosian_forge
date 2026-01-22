import collections.abc
import numbers
import struct
from cmath import isnan
from typing import (
import google.protobuf.message
import numpy as np
from onnx import (
def printable_graph(graph: GraphProto, prefix: str='') -> str:
    """Display a GraphProto as a string.

    Args:
        graph (GraphProto): the graph to display
        prefix (string): prefix of every line

    Returns:
        string
    """
    content = []
    indent = prefix + '  '
    header = ['graph', graph.name]
    initializers = {t.name for t in graph.initializer}
    if len(graph.input):
        header.append('(')
        in_strs = []
        in_with_init_strs = []
        for inp in graph.input:
            if inp.name not in initializers:
                in_strs.append(printable_value_info(inp))
            else:
                in_with_init_strs.append(printable_value_info(inp))
        if in_strs:
            content.append(prefix + ' '.join(header))
            header = []
            for line in in_strs:
                content.append(prefix + '  ' + line)
        header.append(')')
        if in_with_init_strs:
            header.append('optional inputs with matching initializers (')
            content.append(prefix + ' '.join(header))
            header = []
            for line in in_with_init_strs:
                content.append(prefix + '  ' + line)
            header.append(')')
        if len(in_with_init_strs) < len(initializers):
            graph_inputs = {i.name for i in graph.input}
            init_strs = [printable_tensor_proto(i) for i in graph.initializer if i.name not in graph_inputs]
            header.append('initializers (')
            content.append(prefix + ' '.join(header))
            header = []
            for line in init_strs:
                content.append(prefix + '  ' + line)
            header.append(')')
    header.append('{')
    content.append(prefix + ' '.join(header))
    graphs: List[GraphProto] = []
    for node in graph.node:
        contents_subgraphs = printable_node(node, indent, subgraphs=True)
        if not isinstance(contents_subgraphs[1], list):
            raise TypeError(f'contents_subgraphs[1] must be an instance of {list}.')
        content.append(contents_subgraphs[0])
        graphs.extend(contents_subgraphs[1])
    tail = ['return']
    if len(graph.output):
        tail.append(', '.join([f'%{out.name}' for out in graph.output]))
    content.append(indent + ' '.join(tail))
    content.append(prefix + '}')
    for g in graphs:
        content.append('\n' + printable_graph(g))
    return '\n'.join(content)
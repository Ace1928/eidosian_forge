import copy
import io
import errno
import os
import re
import subprocess
import sys
import tempfile
import warnings
import pydot
def parse_node_ref(self, node_str):
    if not isinstance(node_str, str):
        return node_str
    if node_str.startswith('"') and node_str.endswith('"'):
        return node_str
    node_port_idx = node_str.rfind(':')
    if node_port_idx > 0 and node_str[0] == '"' and (node_str[node_port_idx - 1] == '"'):
        return node_str
    if node_port_idx > 0:
        a = node_str[:node_port_idx]
        b = node_str[node_port_idx + 1:]
        node = quote_if_necessary(a)
        node += ':' + quote_if_necessary(b)
        return node
    return node_str
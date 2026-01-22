import sys
from collections import OrderedDict
from distutils.util import strtobool
from aiokeydb.v1.exceptions import ResponseError
from aiokeydb.v1.commands.graph.edge import Edge
from aiokeydb.v1.commands.graph.exceptions import VersionMismatchException
from aiokeydb.v1.commands.graph.node import Node
from aiokeydb.v1.commands.graph.path import Path
@property
def parse_scalar_types(self):
    return {ResultSetScalarTypes.VALUE_NULL: self.parse_null, ResultSetScalarTypes.VALUE_STRING: self.parse_string, ResultSetScalarTypes.VALUE_INTEGER: self.parse_integer, ResultSetScalarTypes.VALUE_BOOLEAN: self.parse_boolean, ResultSetScalarTypes.VALUE_DOUBLE: self.parse_double, ResultSetScalarTypes.VALUE_ARRAY: self.parse_array, ResultSetScalarTypes.VALUE_NODE: self.parse_node, ResultSetScalarTypes.VALUE_EDGE: self.parse_edge, ResultSetScalarTypes.VALUE_PATH: self.parse_path, ResultSetScalarTypes.VALUE_MAP: self.parse_map, ResultSetScalarTypes.VALUE_POINT: self.parse_point, ResultSetScalarTypes.VALUE_UNKNOWN: self.parse_unknown}
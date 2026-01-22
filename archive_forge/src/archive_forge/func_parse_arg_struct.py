from collections import namedtuple
from io import BytesIO
from ..common.utils import struct_parse, bytelist2string, read_blob
from ..common.exceptions import DWARFError
def parse_arg_struct(arg_struct):
    return lambda stream: [struct_parse(arg_struct, stream)]
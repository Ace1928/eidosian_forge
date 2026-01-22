from collections import namedtuple
from io import BytesIO
from ..common.utils import struct_parse, bytelist2string, read_blob
from ..common.exceptions import DWARFError
def parse_op_addr():
    return lambda stream: [struct_parse(structs.Dwarf_target_addr(''), stream)]
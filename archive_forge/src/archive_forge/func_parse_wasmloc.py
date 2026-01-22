from collections import namedtuple
from io import BytesIO
from ..common.utils import struct_parse, bytelist2string, read_blob
from ..common.exceptions import DWARFError
def parse_wasmloc():

    def parse(stream):
        op = struct_parse(structs.Dwarf_uint8(''), stream)
        if 0 <= op <= 2:
            return [op, struct_parse(structs.Dwarf_uleb128(''), stream)]
        elif op == 3:
            return [op, struct_parse(structs.Dwarf_uint32(''), stream)]
        else:
            raise DWARFError('Unknown operation code in DW_OP_WASM_location: %d' % (op,))
    return parse
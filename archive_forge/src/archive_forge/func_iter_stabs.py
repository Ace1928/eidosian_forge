from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def iter_stabs(self):
    """ Yield all stab entries.  Result type is ELFStructs.Elf_Stabs.
        """
    offset = self['sh_offset']
    size = self['sh_size']
    end = offset + size
    while offset < end:
        stabs = struct_parse(self.structs.Elf_Stabs, self.stream, stream_pos=offset)
        stabs['n_offset'] = offset
        offset += self.structs.Elf_Stabs.sizeof()
        self.stream.seek(offset)
        yield stabs
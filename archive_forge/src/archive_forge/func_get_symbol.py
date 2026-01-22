from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def get_symbol(self, n):
    """ Get the symbol at index #n from the table (Symbol object).
            It begins at 1 and not 0 since the first entry is used to
            store the current version of the syminfo table.
        """
    entry_offset = self['sh_offset'] + n * self['sh_entsize']
    entry = struct_parse(self.structs.Elf_Sunw_Syminfo, self.stream, stream_pos=entry_offset)
    name = self.symboltable.get_symbol(n).name
    return Symbol(entry, name)
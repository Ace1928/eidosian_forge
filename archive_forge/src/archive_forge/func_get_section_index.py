from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def get_section_index(self, n):
    """ Get the section header table index for the symbol with index #n.
            The section contains an array of Elf32_word values with one entry
            for every symbol in the associated symbol table.
        """
    return struct_parse(self.elffile.structs.Elf_word(''), self.stream, self['sh_offset'] + n * self['sh_entsize'])
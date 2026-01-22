from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def num_symbols(self):
    """ Number of symbols in the table
        """
    return self['sh_size'] // self['sh_entsize'] - 1
from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
@property
def num_attributes(self):
    """ Number of attributes in the subsubsection.
        """
    return sum((1 for _ in self.iter_attributes())) + 1
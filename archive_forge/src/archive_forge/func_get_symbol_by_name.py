from ..common.exceptions import ELFCompressionError
from ..common.utils import struct_parse, elf_assert, parse_cstring_from_stream
from collections import defaultdict
from .constants import SH_FLAGS
from .notes import iter_notes
import zlib
def get_symbol_by_name(self, name):
    """ Get a symbol(s) by name. Return None if no symbol by the given name
            exists.
        """
    if self._symbol_name_map is None:
        self._symbol_name_map = defaultdict(list)
        for i, sym in enumerate(self.iter_symbols()):
            self._symbol_name_map[sym.name].append(i)
    symnums = self._symbol_name_map.get(name)
    return [self.get_symbol(i) for i in symnums] if symnums else None
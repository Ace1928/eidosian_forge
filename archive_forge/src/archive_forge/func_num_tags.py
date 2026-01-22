import itertools
from collections import defaultdict
from .hash import ELFHashTable, GNUHashTable
from .sections import Section, Symbol
from .enums import ENUM_D_TAG
from .segments import Segment
from .relocation import RelocationTable, RelrRelocationTable
from ..common.exceptions import ELFError
from ..common.utils import elf_assert, struct_parse, parse_cstring_from_stream
def num_tags(self):
    """ Number of dynamic tags in the file, including the DT_NULL tag
        """
    if self._num_tags != -1:
        return self._num_tags
    for n in itertools.count():
        tag = self.get_tag(n)
        if tag.entry.d_tag == 'DT_NULL':
            self._num_tags = n + 1
            return self._num_tags
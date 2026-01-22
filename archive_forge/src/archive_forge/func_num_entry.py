from ..common.utils import struct_parse
from .decoder import EHABIBytecodeDecoder
from .constants import EHABI_INDEX_ENTRY_SIZE
from .structs import EHABIStructs
def num_entry(self):
    """ Number of exception handler entry in the section.
        """
    if self._num_entry is None:
        self._num_entry = self._arm_idx_section['sh_size'] // EHABI_INDEX_ENTRY_SIZE
    return self._num_entry
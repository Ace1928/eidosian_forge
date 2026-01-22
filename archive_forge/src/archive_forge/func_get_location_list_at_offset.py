import os
from collections import namedtuple
from ..common.exceptions import DWARFError
from ..common.utils import struct_parse
from .dwarf_util import _iter_CUs_in_section
def get_location_list_at_offset(self, offset, die=None):
    """ Get a location list at the given offset in the section.
        Passing the die is only neccessary in DWARF5+, for decoding
        location entry encodings that contain references to other sections.
        """
    if self.version >= 5 and die is None:
        raise DWARFError('For this binary, "die" needs to be provided')
    self.stream.seek(offset, os.SEEK_SET)
    return self._parse_location_list_from_stream_v5(die.cu) if self.version >= 5 else self._parse_location_list_from_stream()
from ..construct import CString
from ..common.utils import struct_parse
from .constants import SH_FLAGS
from .notes import iter_notes
def get_interp_name(self):
    """ Obtain the interpreter path used for this ELF file.
        """
    path_offset = self['p_offset']
    return struct_parse(CString('', encoding='utf-8'), self.stream, stream_pos=path_offset)
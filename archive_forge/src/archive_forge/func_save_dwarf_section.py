from contextlib import contextmanager
from .exceptions import ELFParseError, ELFError, DWARFError
from ..construct import ConstructError, ULInt8
import os
def save_dwarf_section(section, filename):
    """Debug helper: dump section contents into a file
    Section is expected to be one of the debug_xxx_sec elements of DWARFInfo
    """
    stream = section.stream
    pos = stream.tell()
    stream.seek(0, os.SEEK_SET)
    section.stream.seek(0)
    with open(filename, 'wb') as file:
        data = stream.read(section.size)
        file.write(data)
    stream.seek(pos, os.SEEK_SET)
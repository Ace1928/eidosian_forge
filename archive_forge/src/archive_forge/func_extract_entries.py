from contextlib import contextmanager
from ctypes import byref, c_longlong, c_size_t, c_void_p
import os
from .ffi import (
from .read import fd_reader, file_reader, memory_reader
def extract_entries(entries, flags=None):
    """Extracts the given archive entries into the current directory.
    """
    if flags is None:
        if os.getcwd() == '/':
            flags = 0
        else:
            flags = PREVENT_ESCAPE
    buff, size, offset = (c_void_p(), c_size_t(), c_longlong())
    buff_p, size_p, offset_p = (byref(buff), byref(size), byref(offset))
    with new_archive_write_disk(flags) as write_p:
        for entry in entries:
            write_header(write_p, entry._entry_p)
            read_p = entry._archive_p
            while 1:
                r = read_data_block(read_p, buff_p, size_p, offset_p)
                if r == ARCHIVE_EOF:
                    break
                write_data_block(write_p, buff, size, offset)
            write_finish_entry(write_p)
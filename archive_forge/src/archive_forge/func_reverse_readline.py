from __future__ import annotations
import bz2
import errno
import gzip
import io
import mmap
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING
def reverse_readline(m_file, blk_size: int=4096, max_mem: int=4000000) -> Generator[str, str, None]:
    """
    Generator method to read a file line-by-line, but backwards. This allows
    one to efficiently get data at the end of a file.

    Based on code by Peter Astrand <astrand@cendio.se>, using modifications by
    Raymond Hettinger and Kevin German.
    http://code.activestate.com/recipes/439045-read-a-text-file-backwards
    -yet-another-implementat/

    Reads file forwards and reverses in memory for files smaller than the
    max_mem parameter, or for gzip files where reverse seeks are not supported.

    Files larger than max_mem are dynamically read backwards.

    Args:
        m_file (File): File stream to read (backwards)
        blk_size (int): The buffer size. Defaults to 4096.
        max_mem (int): The maximum amount of memory to involve in this
            operation. This is used to determine when to reverse a file
            in-memory versus seeking portions of a file. For bz2 files,
            this sets the maximum block size.

    Returns:
        Generator that returns lines from the file. Similar behavior to the
        file.readline() method, except the lines are returned from the back
        of the file.
    """
    is_text = isinstance(m_file, io.TextIOWrapper)
    try:
        file_size = os.path.getsize(m_file.name)
    except AttributeError:
        file_size = max_mem + 1
    if file_size < max_mem or isinstance(m_file, gzip.GzipFile) or os.name == 'nt':
        for line in reversed(m_file.readlines()):
            yield line.rstrip()
    else:
        if isinstance(m_file, bz2.BZ2File):
            blk_size = min(max_mem, file_size)
        buf = ''
        m_file.seek(0, 2)
        lastchar = m_file.read(1) if is_text else m_file.read(1).decode('utf-8')
        trailing_newline = lastchar == '\n'
        while 1:
            newline_pos = buf.rfind('\n')
            pos = m_file.tell()
            if newline_pos != -1:
                line = buf[newline_pos + 1:]
                buf = buf[:newline_pos]
                if pos or newline_pos or trailing_newline:
                    line += '\n'
                yield line
            elif pos:
                toread = min(blk_size, pos)
                m_file.seek(pos - toread, 0)
                if is_text:
                    buf = m_file.read(toread) + buf
                else:
                    buf = m_file.read(toread).decode('utf-8') + buf
                m_file.seek(pos - toread, 0)
                if pos == toread:
                    buf = '\n' + buf
            else:
                return
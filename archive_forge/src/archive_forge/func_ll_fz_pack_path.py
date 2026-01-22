from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_pack_path(pack, path):
    """
    Low-level wrapper for `::fz_pack_path()`.
    Pack a path into the given block.
    To minimise the size of paths, this function allows them to be
    packed into a buffer with other information. Paths can be used
    interchangeably regardless of how they are packed.

    pack: Pointer to a block of data to pack the path into. Should
    be aligned by the caller to the same alignment as required for
    a fz_path pointer.

    path: The path to pack.

    Returns the number of bytes within the block used. Callers can
    access the packed path data by casting the value of pack on
    entry to be a fz_path *.

    Throws exceptions on failure to allocate.

    Implementation details: Paths can be 'unpacked', 'flat', or
    'open'. Standard paths, as created are 'unpacked'. Paths
    will be packed as 'flat', unless they are too large
    (where large indicates that they exceed some private
    implementation defined limits, currently including having
    more than 256 coordinates or commands).

    Large paths are 'open' packed as a header into the given block,
    plus pointers to other data blocks.

    Users should not have to care about whether paths are 'open'
    or 'flat' packed. Simply pack a path (if required), and then
    forget about the details.
    """
    return _mupdf.ll_fz_pack_path(pack, path)
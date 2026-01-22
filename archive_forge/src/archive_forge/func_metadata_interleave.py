import warnings
import functools
import locale
import weakref
import ctypes
import html
import textwrap
import llvmlite.binding as ll
import llvmlite.ir as llvmir
from abc import abstractmethod, ABCMeta
from numba.core import utils, config, cgutils
from numba.core.llvm_bindings import create_pass_manager_builder
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
from numba.core.runtime import rtsys
from numba.core.compiler_lock import require_global_compiler_lock
from numba.core.errors import NumbaInvalidConfigWarning
from numba.misc.inspection import disassemble_elf_to_cfg
from numba.misc.llvm_pass_timings import PassTimingsCollection
def metadata_interleave(l, new_lines):
    """
                Search line `l` for metadata associated with python or line info
                and inject it into `new_lines` if requested.
                """
    matched = metadata_marker.match(l)
    if matched is not None:
        g = matched.groups()
        if g is not None:
            assert len(g) == 1, g
            marker = g[0]
            debug_data = md.get(marker, None)
            if debug_data is not None:
                ld = location_entry.match(debug_data)
                if ld is not None:
                    assert len(ld.groups()) == 2, ld
                    line, col = ld.groups()
                    if line != cur_line or col != cur_col:
                        if _interleave.lineinfo:
                            mfmt = 'Marker %s, Line %s, column %s'
                            mark_line = mfmt % (marker, line, col)
                            ln = fmt.format(cs['marker'], col_span, clean(mark_line))
                            new_lines.append(ln)
                        if _interleave.python:
                            lidx = int(line) - (firstlineno + 1)
                            source_line = src_code[lidx + 1]
                            ln = fmt.format(cs['python'], col_span, clean(source_line))
                            new_lines.append(ln)
                        return (line, col)
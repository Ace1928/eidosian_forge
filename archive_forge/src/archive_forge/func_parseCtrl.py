import functools
import os
import re
import subprocess
import tempfile
from ..common.backend import path_to_cuobjdump, path_to_nvdisasm
def parseCtrl(sline):
    enc = int(SLINE_RE.match(sline).group(1), 16)
    stall = enc >> 41 & 15
    yld = enc >> 45 & 1
    wrtdb = enc >> 46 & 7
    readb = enc >> 49 & 7
    watdb = enc >> 52 & 63
    yld_str = 'Y' if yld == 0 else '-'
    wrtdb_str = '-' if wrtdb == 7 else str(wrtdb)
    readb_str = '-' if readb == 7 else str(readb)
    watdb_str = '--' if watdb == 0 else f'{watdb:02d}'
    return f'{watdb_str}:{readb_str}:{wrtdb_str}:{yld_str}:{stall:x}'
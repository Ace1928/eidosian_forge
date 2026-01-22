import functools
import os
import re
import subprocess
import tempfile
from ..common.backend import path_to_cuobjdump, path_to_nvdisasm
def processSassLines(fline, sline, labels):
    asm = FLINE_RE.match(fline).group(1)
    if asm.endswith(' ;'):
        asm = asm[:-2] + ';'
    ctrl = parseCtrl(sline)
    if BRA_RE.match(asm) is not None:
        target = int(BRA_RE.match(asm).group(2), 16)
        if target in labels:
            pass
        else:
            labels[target] = len(labels)
    return (f'{ctrl}', f'{asm}')
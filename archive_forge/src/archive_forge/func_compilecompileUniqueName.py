from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval
from . import DefaultTable
def compilecompileUniqueName(self, name, length):
    nameLen = len(name)
    if length <= nameLen:
        name = name[:length - 1] + '\x00'
    else:
        name += (nameLen - length) * '\x00'
    return name
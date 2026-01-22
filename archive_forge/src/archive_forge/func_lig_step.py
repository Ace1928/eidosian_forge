from types import SimpleNamespace
from fontTools.misc.sstruct import calcsize, unpack, unpack2
def lig_step(i):
    return 4 * (lig_kern_base + i)
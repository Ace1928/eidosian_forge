from types import SimpleNamespace
from fontTools.misc.sstruct import calcsize, unpack, unpack2
def lig_kern_command(i):
    command = SimpleNamespace()
    unpack2(LIG_KERN_COMMAND, data[i:], command)
    return command
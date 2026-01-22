from . import __version__
from .auxfuncs import (
from . import capi_maps
from . import func2subr
from .crackfortran import rmbadname
def findcommonblocks(block, top=1):
    ret = []
    if hascommon(block):
        for key, value in block['common'].items():
            vars_ = {v: block['vars'][v] for v in value}
            ret.append((key, value, vars_))
    elif hasbody(block):
        for b in block['body']:
            ret = ret + findcommonblocks(b, 0)
    if top:
        tret = []
        names = []
        for t in ret:
            if t[0] not in names:
                names.append(t[0])
                tret.append(t)
        return tret
    return ret
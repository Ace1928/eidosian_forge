from . import __version__
from .auxfuncs import (
from . import cfuncs
def buildcallbacks(m):
    cb_map[m['name']] = []
    for bi in m['body']:
        if bi['block'] == 'interface':
            for b in bi['body']:
                if b:
                    buildcallback(b, m['name'])
                else:
                    errmess('warning: empty body for %s\n' % m['name'])
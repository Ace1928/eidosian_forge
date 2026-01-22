import pkgutil
import re
from jsbeautifier.unpackers import evalbased
def getunpackers():
    """Scans the unpackers dir, finds unpackers and add them to UNPACKERS list.
    An unpacker will be loaded only if it is a valid python module (name must
    adhere to naming conventions) and it is not blacklisted (i.e. inserted
    into BLACKLIST."""
    path = __path__
    prefix = __name__ + '.'
    unpackers = []
    interface = ['unpack', 'detect', 'PRIORITY']
    for _importer, modname, _ispkg in pkgutil.iter_modules(path, prefix):
        if 'tests' not in modname and modname not in BLACKLIST:
            try:
                module = __import__(modname, fromlist=interface)
            except ImportError:
                raise UnpackingError('Bad unpacker: %s' % modname)
            else:
                unpackers.append(module)
    return sorted(unpackers, key=lambda mod: mod.PRIORITY)
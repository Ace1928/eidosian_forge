import os
import sys
from os.path import pardir, realpath
def get_preferred_scheme(key):
    if key == 'prefix' and sys.prefix != sys.base_prefix:
        return 'venv'
    scheme = _get_preferred_schemes()[key]
    if scheme not in _INSTALL_SCHEMES:
        raise ValueError(f'{key!r} returned {scheme!r}, which is not a valid scheme on this platform')
    return scheme
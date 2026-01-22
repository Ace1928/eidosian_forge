from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
@classmethod
def priority(klass, transport):
    if 'hg' in transport.base:
        return 90
    return 99
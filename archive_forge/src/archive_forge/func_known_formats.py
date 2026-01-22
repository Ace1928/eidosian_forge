from ... import version_info  # noqa: F401
from ... import controldir, errors
from ... import transport as _mod_transport
@classmethod
def known_formats(cls):
    return [SmartHgDirFormat()]
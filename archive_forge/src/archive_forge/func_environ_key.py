imports the variables from this location, and once that 
import occurs the flags should no longer be changed.
from OpenGL.version import __version__
import os
from OpenGL.plugins import PlatformPlugin, FormatHandler
import sys
def environ_key(name, default):
    composed = 'PYOPENGL_%s' % name.upper()
    if composed in os.environ:
        value = os.environ[composed]
        if value.lower() in ('1', 'true'):
            return True
        else:
            return False
    return os.environ.get(composed, default)
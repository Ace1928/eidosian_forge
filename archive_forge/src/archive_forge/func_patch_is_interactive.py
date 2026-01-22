import sys
from _pydev_bundle import pydev_log
def patch_is_interactive():
    """ Patch matplotlib function 'use' """
    matplotlib = sys.modules['matplotlib']

    def patched_is_interactive():
        return matplotlib.rcParams['interactive']
    matplotlib.real_is_interactive = matplotlib.is_interactive
    matplotlib.is_interactive = patched_is_interactive
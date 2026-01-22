import os
import sys
import warnings
from os.path import basename, dirname, exists, join, splitext
from pygame.font import Font
def initsysfonts():
    """
    Initialise the sysfont module, called once. Locates the installed fonts
    and creates some aliases for common font categories.

    Has different initialisation functions for different platforms.
    """
    global is_init
    if is_init:
        return
    if sys.platform == 'win32':
        fonts = initsysfonts_win32()
    elif sys.platform == 'darwin':
        fonts = initsysfonts_darwin()
    else:
        fonts = initsysfonts_unix()
    Sysfonts.update(fonts)
    create_aliases()
    is_init = True
from collections import OrderedDict
import numpy as np
import os
import re
import struct
import sys
import time
import logging
def resource_dirs():
    """resource_dirs()

    Get a list of directories where imageio resources may be located.
    The first directory in this list is the "resources" directory in
    the package itself. The second directory is the appdata directory
    (~/.imageio on Linux). The list further contains the application
    directory (for frozen apps), and may include additional directories
    in the future.
    """
    dirs = [resource_package_dir()]
    try:
        dirs.append(appdata_dir('imageio'))
    except Exception:
        pass
    if getattr(sys, 'frozen', None):
        dirs.append(os.path.abspath(os.path.dirname(sys.executable)))
    elif sys.path and sys.path[0]:
        dirs.append(os.path.abspath(sys.path[0]))
    return dirs
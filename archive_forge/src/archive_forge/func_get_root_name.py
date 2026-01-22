import errno
import os
import sys
import time
from . import archive, errors, osutils, trace
def get_root_name(dest):
    """Get just the root name for an export.

    """
    global _exporter_extensions
    if dest == '-':
        return ''
    dest = os.path.basename(dest)
    for ext in archive.format_registry.extensions:
        if dest.endswith(ext):
            return dest[:-len(ext)]
    return dest
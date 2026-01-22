from __future__ import annotations
import sys
import traceback
def default_encoding(file=None):
    """Get default encoding."""
    file = file or get_default_encoding_file()
    return getattr(file, 'encoding', None) or sys.getfilesystemencoding()
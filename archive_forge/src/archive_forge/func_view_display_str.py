import re
from . import errors, osutils, transport
def view_display_str(view_files, encoding=None):
    """Get the display string for a list of view files.

    Args:
      view_files: the list of file names
      encoding: the encoding to display the files in
    """
    if encoding is None:
        return ', '.join(view_files)
    else:
        return ', '.join([v.encode(encoding, 'replace') for v in view_files])
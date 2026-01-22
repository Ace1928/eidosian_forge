import atexit
import os
import signal
import sys
import ovs.vlog
def remove_file_to_unlink(file):
    """Unregisters 'file' from being unlinked when the program terminates via
    sys.exit() or a fatal signal."""
    if file in _files:
        del _files[file]
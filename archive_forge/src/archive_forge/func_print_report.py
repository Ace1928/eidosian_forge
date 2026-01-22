from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def print_report(self, file=sys.stdout):
    """Prints a report of line memory profiling."""
    line = '_root (%s, %s)\n' % self._root.humanized_bytes()
    file.write(line)
    for child in self._root.children:
        self._print_frame(child, depth=1, file=file)
    file.flush()
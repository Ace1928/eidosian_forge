import bisect
from _pydevd_bundle.pydevd_constants import NULL, KeyifyList
import pydevd_file_utils
def map_to_server(self, absolute_filename, lineno):
    """
        Convert something as 'file1.py' at line 10 to '<ipython-cell-xxx>' at line 2.

        Note that the name should be already normalized at this point.
        """
    absolute_normalized_filename = pydevd_file_utils.normcase(absolute_filename)
    changed = False
    mappings = self._mappings_to_server.get(absolute_normalized_filename)
    if mappings:
        i = bisect.bisect(KeyifyList(mappings, lambda entry: entry.line), lineno)
        if i >= len(mappings):
            i -= 1
        if i == 0:
            entry = mappings[i]
        else:
            entry = mappings[i - 1]
        if not entry.contains_line(lineno):
            entry = mappings[i]
            if not entry.contains_line(lineno):
                entry = None
        if entry is not None:
            lineno = entry.runtime_line + (lineno - entry.line)
            absolute_filename = entry.runtime_source
            changed = True
    return (absolute_filename, lineno, changed)
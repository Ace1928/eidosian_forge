import difflib
import patiencediff
from merge3 import Merge3
from ... import debug, merge, osutils
from ...trace import mutter
def changelog_entries(lines):
    """Return a list of changelog entries.

    :param lines: lines of a changelog file.
    :returns: list of entries.  Each entry is a tuple of lines.
    """
    entries = []
    for line in lines:
        if line[0] not in (' ', '\t', '\n'):
            entries.append([line])
        else:
            try:
                entry = entries[-1]
            except IndexError:
                entries.append([])
                entry = entries[-1]
            entry.append(line)
    return list(map(tuple, entries))
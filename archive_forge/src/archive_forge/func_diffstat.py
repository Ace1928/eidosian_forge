import re
import sys
from typing import List, Optional, Tuple
def diffstat(lines, max_width=80):
    """Generate summary statistics from a git style diff ala
       (git diff tag1 tag2 --stat).

    Args:
      lines: list of byte string "lines" from the diff to be parsed
      max_width: maximum line length for generating the summary
                 statistics (default 80)
    Returns: A byte string that lists the changed files with change
             counts and histogram.
    """
    names, nametypes, counts = _parse_patch(lines)
    insert = []
    delete = []
    namelen = 0
    maxdiff = 0
    for i, filename in enumerate(names):
        i, d = counts[i]
        insert.append(i)
        delete.append(d)
        namelen = max(namelen, len(filename))
        maxdiff = max(maxdiff, i + d)
    output = b''
    statlen = len(str(maxdiff))
    for i, n in enumerate(names):
        binaryfile = nametypes[i]
        format = b' %-' + str(namelen).encode('ascii') + b's | %' + str(statlen).encode('ascii') + b's %s\n'
        binformat = b' %-' + str(namelen).encode('ascii') + b's | %s\n'
        if not binaryfile:
            hist = b''
            width = len(format % (b'', b'', b''))
            histwidth = max(2, max_width - width)
            if maxdiff < histwidth:
                hist = b'+' * insert[i] + b'-' * delete[i]
            else:
                iratio = float(insert[i]) / maxdiff * histwidth
                dratio = float(delete[i]) / maxdiff * histwidth
                iwidth = dwidth = 0
                if insert[i] > 0:
                    iwidth = int(iratio)
                    if iwidth == 0 and 0 < iratio < 1:
                        iwidth = 1
                if delete[i] > 0:
                    dwidth = int(dratio)
                    if dwidth == 0 and 0 < dratio < 1:
                        dwidth = 1
                hist = b'+' * int(iwidth) + b'-' * int(dwidth)
            output += format % (bytes(names[i]), str(insert[i] + delete[i]).encode('ascii'), hist)
        else:
            output += binformat % (bytes(names[i]), b'Bin')
    output += b' %d files changed, %d insertions(+), %d deletions(-)' % (len(names), sum(insert), sum(delete))
    return output
import os
from fnmatch import fnmatch
from datetime import datetime
import operator
import re
def report_expected_diffs(diffs, colorize=False):
    """
    Takes the output of compare_expected, and returns a string
    description of the differences.
    """
    if not diffs:
        return 'No differences'
    diffs = diffs.items()
    diffs.sort()
    s = []
    last = ''
    for path, desc in diffs:
        t = _space_prefix(last, path, indent=4, include_sep=False)
        if colorize:
            t = color_line(t, 11)
        last = path
        if len(desc.splitlines()) > 1:
            cur_indent = len(re.search('^[ ]*', t).group(0))
            desc = indent(cur_indent + 2, desc)
            if colorize:
                t += '\n'
                for line in desc.splitlines():
                    if line.strip().startswith('+'):
                        line = color_line(line, 10)
                    elif line.strip().startswith('-'):
                        line = color_line(line, 9)
                    else:
                        line = color_line(line, 14)
                    t += line + '\n'
            else:
                t += '\n' + desc
        else:
            t += ' ' + desc
        s.append(t)
    s.append('Files with differences: %s' % len(diffs))
    return '\n'.join(s)
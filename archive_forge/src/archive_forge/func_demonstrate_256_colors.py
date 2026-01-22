import functools
import getopt
import pipes
import subprocess
import sys
from humanfriendly import (
from humanfriendly.tables import format_pretty_table, format_smart_table
from humanfriendly.terminal import (
from humanfriendly.terminal.spinners import Spinner
def demonstrate_256_colors(i, j, group=None):
    """Demonstrate 256 color mode support."""
    label = '256 color mode'
    if group:
        label += ' (%s)' % group
    output('\n' + ansi_wrap('%s:' % label, bold=True))
    single_line = ''.join((' ' + ansi_wrap(str(n), color=n) for n in range(i, j + 1)))
    lines, columns = find_terminal_size()
    if columns >= len(ansi_strip(single_line)):
        output(single_line)
    else:
        width = len(str(j)) + 1
        colors_per_line = int(columns / width)
        colors = [ansi_wrap(str(n).rjust(width), color=n) for n in range(i, j + 1)]
        blocks = [colors[n:n + colors_per_line] for n in range(0, len(colors), colors_per_line)]
        output('\n'.join((''.join(b) for b in blocks)))
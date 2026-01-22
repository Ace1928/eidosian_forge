import operator
from ... import (branch, commands, config, errors, option, trace, tsort, ui,
from ...revision import NULL_REVISION
from .classify import classify_delta
def print_section(name, lst):
    if len(lst) == 0:
        return
    to_file.write('%s:\n' % name)
    for name in lst:
        to_file.write('%s\n' % name)
    to_file.write('\n')
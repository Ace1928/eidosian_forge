from __future__ import division
import re
import stat
from .helpers import (
def dump_str(self, names=None, child_lists=None, verbose=False):
    result = [ImportCommand.dump_str(self, names, verbose=verbose)]
    for f in self.iter_files():
        if child_lists is None:
            continue
        try:
            child_names = child_lists[f.name]
        except KeyError:
            continue
        result.append('\t%s' % f.dump_str(child_names, verbose=verbose))
    return '\n'.join(result)
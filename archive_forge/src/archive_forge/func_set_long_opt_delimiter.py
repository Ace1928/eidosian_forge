import sys, os
import textwrap
def set_long_opt_delimiter(self, delim):
    if delim not in ('=', ' '):
        raise ValueError('invalid metavar delimiter for long options: %r' % delim)
    self._long_opt_fmt = '%s' + delim + '%s'
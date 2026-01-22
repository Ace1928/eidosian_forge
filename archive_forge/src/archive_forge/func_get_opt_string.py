import sys, os
import textwrap
def get_opt_string(self):
    if self._long_opts:
        return self._long_opts[0]
    else:
        return self._short_opts[0]
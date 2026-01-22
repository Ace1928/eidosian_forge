import os
import io
import re
import sys
import cmd
import bdb
import dis
import code
import glob
import pprint
import signal
import inspect
import tokenize
import functools
import traceback
import linecache
from typing import Union
def do_ignore(self, arg):
    """ignore bpnumber [count]
        Set the ignore count for the given breakpoint number.  If
        count is omitted, the ignore count is set to 0.  A breakpoint
        becomes active when the ignore count is zero.  When non-zero,
        the count is decremented each time the breakpoint is reached
        and the breakpoint is not disabled and any associated
        condition evaluates to true.
        """
    args = arg.split()
    try:
        count = int(args[1].strip())
    except:
        count = 0
    try:
        bp = self.get_bpbynumber(args[0].strip())
    except IndexError:
        self.error('Breakpoint number expected')
    except ValueError as err:
        self.error(err)
    else:
        bp.ignore = count
        if count > 0:
            if count > 1:
                countstr = '%d crossings' % count
            else:
                countstr = '1 crossing'
            self.message('Will ignore next %s of breakpoint %d.' % (countstr, bp.number))
        else:
            self.message('Will stop next time breakpoint %d is reached.' % bp.number)
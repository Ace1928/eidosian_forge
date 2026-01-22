import itertools
import unittest
import sys
from autopage.tests import isolation
import typing
import autopage
def from_stdin() -> int:
    ap = autopage.AutoPager(pager_command=autopage.command.Less(), line_buffering=autopage.line_buffer_from_input())
    with ap as out:
        try:
            for line in sys.stdin:
                print(line.rstrip(), file=out)
        except KeyboardInterrupt:
            pass
    return ap.exit_code()
import sys, os
import textwrap
def set_conflict_handler(self, handler):
    if handler not in ('error', 'resolve'):
        raise ValueError('invalid conflict_resolution value %r' % handler)
    self.conflict_handler = handler
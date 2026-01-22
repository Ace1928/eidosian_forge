from __future__ import absolute_import
import cython
from . import Errors
from .Regexps import BOL, EOL, EOF
def produce(self, value, text=None):
    """
        Called from an action procedure, causes |value| to be returned
        as the token value from read(). If |text| is supplied, it is
        returned in place of the scanned text.

        produce() can be called more than once during a single call to an action
        procedure, in which case the tokens are queued up and returned one
        at a time by subsequent calls to read(), until the queue is empty,
        whereupon scanning resumes.
        """
    if text is None:
        text = self.text
    self.queue.append(((value, text), self.current_scanner_position_tuple))
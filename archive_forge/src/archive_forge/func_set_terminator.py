import asyncore
from collections import deque
from warnings import _deprecated
def set_terminator(self, term):
    """Set the input delimiter.

        Can be a fixed string of any length, an integer, or None.
        """
    if isinstance(term, str) and self.use_encoding:
        term = bytes(term, self.encoding)
    elif isinstance(term, int) and term < 0:
        raise ValueError('the number of received bytes must be positive')
    self.terminator = term
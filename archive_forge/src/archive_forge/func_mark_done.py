import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def mark_done(self, top, left, bottom, right):
    """For keeping track of how much of each text column has been seen."""
    before = top - 1
    after = bottom - 1
    for col in range(left, right):
        assert self.done[col] == before
        self.done[col] = after
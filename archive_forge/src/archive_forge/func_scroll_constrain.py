import codecs
import copy
import sys
import warnings
def scroll_constrain(self):
    """This keeps the scroll region within the screen region."""
    if self.scroll_row_start <= 0:
        self.scroll_row_start = 1
    if self.scroll_row_end > self.rows:
        self.scroll_row_end = self.rows
import codecs
import copy
import sys
import warnings
def scroll_screen_rows(self, rs, re):
    """Enable scrolling from row {start} to row {end}."""
    self.scroll_row_start = rs
    self.scroll_row_end = re
    self.scroll_constrain()
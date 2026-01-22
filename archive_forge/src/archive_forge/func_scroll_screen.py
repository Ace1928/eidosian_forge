import codecs
import copy
import sys
import warnings
def scroll_screen(self):
    """Enable scrolling for entire display."""
    self.scroll_row_start = 1
    self.scroll_row_end = self.rows
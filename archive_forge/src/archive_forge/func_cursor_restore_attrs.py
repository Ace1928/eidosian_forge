import codecs
import copy
import sys
import warnings
def cursor_restore_attrs(self):
    """Restores cursor position after a Save Cursor."""
    self.cursor_home(self.cur_saved_r, self.cur_saved_c)
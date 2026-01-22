import codecs
import copy
import sys
import warnings
def cursor_up_reverse(self):
    old_r = self.cur_r
    self.cursor_up()
    if old_r == self.cur_r:
        self.scroll_up()
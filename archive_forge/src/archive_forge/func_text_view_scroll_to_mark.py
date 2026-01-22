import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def text_view_scroll_to_mark(self, mark, within_margin, use_align=False, xalign=0.5, yalign=0.5):
    return orig_text_view_scroll_to_mark(self, mark, within_margin, use_align, xalign, yalign)
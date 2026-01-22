from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING, Optional
from pyglet import event
from pyglet.text import runlist
def get_font_runs(self, dpi=None):
    return _FontStyleRunsRangeIterator(self.get_style_runs('font_name'), self.get_style_runs('font_size'), self.get_style_runs('bold'), self.get_style_runs('italic'), self.get_style_runs('stretch'), dpi)
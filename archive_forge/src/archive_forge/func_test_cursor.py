from .lib import TestBase, FileCreator
from smmap.mman import (
from smmap.util import align_to_mmap
from random import randint
from time import time
import os
import sys
from copy import copy
def test_cursor(self):
    with FileCreator(self.k_window_test_size, 'cursor_test') as fc:
        man = SlidingWindowMapManager()
        ci = WindowCursor(man)
        assert not ci.is_valid()
        assert not ci.is_associated()
        assert ci.size() == 0
        cv = man.make_cursor(fc.path)
        assert not cv.is_valid()
        assert cv.is_associated()
        assert cv.file_size() == fc.size
        assert cv.path() == fc.path
    cio = copy(cv)
    assert not cio.is_valid() and cio.is_associated()
    assert not ci.is_associated()
    ci.assign(cv)
    assert not ci.is_valid() and ci.is_associated()
    cv.unuse_region()
    cv.unuse_region()
    cv._destroy()
    WindowCursor(man)._destroy()
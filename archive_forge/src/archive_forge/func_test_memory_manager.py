from .lib import TestBase, FileCreator
from smmap.mman import (
from smmap.util import align_to_mmap
from random import randint
from time import time
import os
import sys
from copy import copy
def test_memory_manager(self):
    slide_man = SlidingWindowMapManager()
    static_man = StaticWindowMapManager()
    for man in (static_man, slide_man):
        assert man.num_file_handles() == 0
        assert man.num_open_files() == 0
        winsize_cmp_val = 0
        if isinstance(man, StaticWindowMapManager):
            winsize_cmp_val = -1
        assert man.window_size() > winsize_cmp_val
        assert man.mapped_memory_size() == 0
        assert man.max_mapped_memory_size() > 0
        man._collect_lru_region(0)
        man._collect_lru_region(10)
        assert man._collect_lru_region(sys.maxsize) == 0
        with FileCreator(self.k_window_test_size, 'manager_test') as fc:
            fd = os.open(fc.path, os.O_RDONLY)
            try:
                for item in (fc.path, fd):
                    c = man.make_cursor(item)
                    assert c.path_or_fd() is item
                    assert c.use_region(10, 10).is_valid()
                    assert c.ofs_begin() == 10
                    assert c.size() == 10
                    with open(fc.path, 'rb') as fp:
                        assert c.buffer()[:] == fp.read(20)[10:]
                if isinstance(item, int):
                    self.assertRaises(ValueError, c.path)
                else:
                    self.assertRaises(ValueError, c.fd)
            finally:
                os.close(fd)
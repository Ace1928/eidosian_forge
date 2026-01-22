from .lib import TestBase, FileCreator
from smmap.util import (
import os
import sys
def test_region(self):
    with FileCreator(self.k_window_test_size, 'window_test') as fc:
        half_size = fc.size // 2
        rofs = align_to_mmap(4200, False)
        rfull = MapRegion(fc.path, 0, fc.size)
        rhalfofs = MapRegion(fc.path, rofs, fc.size)
        rhalfsize = MapRegion(fc.path, 0, half_size)
        assert rfull.ofs_begin() == 0 and rfull.size() == fc.size
        assert rfull.ofs_end() == fc.size
        assert rhalfofs.ofs_begin() == rofs and rhalfofs.size() == fc.size - rofs
        assert rhalfsize.ofs_begin() == 0 and rhalfsize.size() == half_size
        assert rfull.includes_ofs(0) and rfull.includes_ofs(fc.size - 1) and rfull.includes_ofs(half_size)
        assert not rfull.includes_ofs(-1) and (not rfull.includes_ofs(sys.maxsize))
    assert rfull.client_count() == 1
    rfull2 = rfull
    assert rfull.client_count() == 1, 'no auto-counting'
    w = MapWindow.from_region(rfull)
    assert w.ofs == rfull.ofs_begin() and w.ofs_end() == rfull.ofs_end()
import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
def test_link_creation_tracking(self):
    """
        tests the link creation order set/get
        """
    gcid = h5p.create(h5p.GROUP_CREATE)
    gcid.set_link_creation_order(0)
    self.assertEqual(0, gcid.get_link_creation_order())
    flags = h5p.CRT_ORDER_TRACKED | h5p.CRT_ORDER_INDEXED
    gcid.set_link_creation_order(flags)
    self.assertEqual(flags, gcid.get_link_creation_order())
    fcpl = h5p.create(h5p.FILE_CREATE)
    fcpl.set_link_creation_order(flags)
    self.assertEqual(flags, fcpl.get_link_creation_order())
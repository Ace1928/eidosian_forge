import numpy as np
import os
import os.path
import sys
from tempfile import mkdtemp
from collections.abc import MutableMapping
from .common import ut, TestCase
import h5py
from h5py import File, Group, SoftLink, HardLink, ExternalLink
from h5py import Dataset, Datatype
from h5py import h5t
from h5py._hl.compat import filename_encode
def test_get_link_class(self):
    """ Get link classes """
    default = object()
    sl = SoftLink('/mongoose')
    el = ExternalLink('somewhere.hdf5', 'mongoose')
    self.f.create_group('hard')
    self.f['soft'] = sl
    self.f['external'] = el
    out_hl = self.f.get('hard', default, getlink=True, getclass=True)
    out_sl = self.f.get('soft', default, getlink=True, getclass=True)
    out_el = self.f.get('external', default, getlink=True, getclass=True)
    self.assertEqual(out_hl, HardLink)
    self.assertEqual(out_sl, SoftLink)
    self.assertEqual(out_el, ExternalLink)
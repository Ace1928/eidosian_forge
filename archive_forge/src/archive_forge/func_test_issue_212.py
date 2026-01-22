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
def test_issue_212(self):
    """ Issue 212

        Fails with:

        AttributeError: 'SharedConfig' object has no attribute 'lapl'
        """

    def closer(x):

        def w():
            try:
                if x:
                    x.close()
            except IOError:
                pass
        return w
    orig_name = self.mktemp()
    new_name = self.mktemp()
    f = File(orig_name, 'w')
    self.addCleanup(closer(f))
    f.create_group('a')
    f.close()
    g = File(new_name, 'w')
    self.addCleanup(closer(g))
    g['link'] = ExternalLink(orig_name, '/')
    g.close()
    h = File(new_name, 'r')
    self.addCleanup(closer(h))
    self.assertIsInstance(h['link']['a'], Group)
from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds_path_no_subdir(self):
    args = [('dsname', ['', 'x.vmdk']), ('dsname', ['x.vmdk'])]
    canonical_p = datastore.DatastorePath('dsname', 'x.vmdk')
    self.assertEqual('[dsname] x.vmdk', str(canonical_p))
    self.assertEqual('', canonical_p.dirname)
    self.assertEqual('x.vmdk', canonical_p.basename)
    self.assertEqual('x.vmdk', canonical_p.rel_path)
    for t in args:
        p = datastore.DatastorePath(t[0], *t[1])
        self.assertEqual(str(canonical_p), str(p))
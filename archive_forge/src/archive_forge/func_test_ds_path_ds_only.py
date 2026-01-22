from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds_path_ds_only(self):
    args = [('dsname', []), ('dsname', ['']), ('dsname', ['', ''])]
    canonical_p = datastore.DatastorePath('dsname')
    self.assertEqual('[dsname]', str(canonical_p))
    self.assertEqual('', canonical_p.rel_path)
    self.assertEqual('', canonical_p.basename)
    self.assertEqual('', canonical_p.dirname)
    for t in args:
        p = datastore.DatastorePath(t[0], *t[1])
        self.assertEqual(str(canonical_p), str(p))
        self.assertEqual(canonical_p.rel_path, p.rel_path)
from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds_path_equivalence(self):
    args = [('dsname', ['a/b/c/', 'x.vmdk']), ('dsname', ['a/', 'b/c/', 'x.vmdk']), ('dsname', ['a', 'b', 'c', 'x.vmdk']), ('dsname', ['a/b/c', 'x.vmdk'])]
    canonical_p = datastore.DatastorePath('dsname', 'a/b/c', 'x.vmdk')
    for t in args:
        p = datastore.DatastorePath(t[0], *t[1])
        self.assertEqual(str(canonical_p), str(p))
        self.assertEqual(canonical_p.datastore, p.datastore)
        self.assertEqual(canonical_p.rel_path, p.rel_path)
        self.assertEqual(str(canonical_p.parent), str(p.parent))
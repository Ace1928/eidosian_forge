from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds_path_no_ds_name(self):
    bad_args = [('', ['a/b/c', 'file.iso']), (None, ['a/b/c', 'file.iso'])]
    for t in bad_args:
        self.assertRaises(ValueError, datastore.DatastorePath, t[0], *t[1])
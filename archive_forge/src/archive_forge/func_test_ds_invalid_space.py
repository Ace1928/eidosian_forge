from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_ds_invalid_space(self):
    self.assertRaises(ValueError, datastore.Datastore, 'fake_ref', 'ds_name', 1 * units.Gi, 2 * units.Gi)
    self.assertRaises(ValueError, datastore.Datastore, 'fake_ref', 'ds_name', None, 2 * units.Gi)
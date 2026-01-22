from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_connected_hosts_in_maintenance(self):
    hosts = self._test_get_connected_hosts(True)
    self.assertEqual(0, len(hosts))
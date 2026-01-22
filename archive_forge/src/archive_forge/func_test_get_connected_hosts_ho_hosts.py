from unittest import mock
from oslo_utils import units
import urllib.parse as urlparse
from oslo_vmware import constants
from oslo_vmware.objects import datastore
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_connected_hosts_ho_hosts(self):
    hosts = self._test_get_connected_hosts(False, False)
    self.assertEqual(0, len(hosts))
import os
from unittest import mock
import urllib.parse as urlparse
import urllib.request as urllib
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_profiles_by_empty_ids(self):
    session = mock.Mock()
    self.assertEqual([], pbm.get_profiles_by_ids(session, []))
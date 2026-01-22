import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_vc_version(self):
    session = mock.Mock()
    expected_version = '6.0.1'
    session.vim.service_content.about.version = expected_version
    version = vim_util.get_vc_version(session)
    self.assertEqual(expected_version, version)
    expected_version = '5.5'
    session.vim.service_content.about.version = expected_version
    version = vim_util.get_vc_version(session)
    self.assertEqual(expected_version, version)
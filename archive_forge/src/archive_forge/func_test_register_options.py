from unittest import mock
from oslotest import base as test_base
from testtools import matchers
from oslo_log import versionutils
@mock.patch.object(versionutils.CONF, 'register_opts')
def test_register_options(self, mock_register_opts):
    versionutils.register_options()
    mock_register_opts.assert_called_once_with(versionutils.deprecated_opts)
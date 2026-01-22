import builtins
import gettext
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
from oslo_i18n import _gettextutils
from oslo_i18n import _lazy
from oslo_i18n import _message
def test_gettext_install_looks_up_localedir(self):
    with mock.patch('os.environ.get') as environ_get:
        with mock.patch('gettext.install'):
            environ_get.return_value = '/foo/bar'
            _gettextutils.install('blaa')
            environ_get.assert_has_calls([mock.call('BLAA_LOCALEDIR')])
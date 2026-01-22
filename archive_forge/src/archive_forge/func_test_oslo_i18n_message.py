from unittest import mock
from oslo_i18n import fixture as oslo_i18n_fixture
from oslotest import base as test_base
from oslo_utils import encodeutils
def test_oslo_i18n_message(self):
    exc = oslo_i18n_fixture.Translation().lazy('test')
    self.assertEqual(encodeutils.exception_to_unicode(exc), 'test')
from unittest import mock
from oslo_i18n import fixture as oslo_i18n_fixture
from oslotest import base as test_base
from oslo_utils import encodeutils
def test_safe_encode_force_incoming_utf8_to_ascii(self):
    self.assertEqual('niÃ±o'.encode('latin-1'), encodeutils.safe_encode('niÃ±o'.encode('latin-1'), incoming='ascii'))
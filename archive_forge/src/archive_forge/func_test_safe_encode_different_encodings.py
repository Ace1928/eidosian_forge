from unittest import mock
from oslo_i18n import fixture as oslo_i18n_fixture
from oslotest import base as test_base
from oslo_utils import encodeutils
def test_safe_encode_different_encodings(self):
    text = 'fooÃ±bar'
    result = encodeutils.safe_encode(text=text, incoming='utf-8', encoding='iso-8859-1')
    self.assertNotEqual(text, result)
    self.assertNotEqual('fooñbar'.encode('latin-1'), result)
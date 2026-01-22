from unittest import mock
from oslo_i18n import fixture as oslo_i18n_fixture
from oslotest import base as test_base
from oslo_utils import encodeutils
def test_safe_encode_int_instead_of_text(self):
    self.assertRaises(TypeError, encodeutils.safe_encode, 1)
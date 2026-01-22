import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
@mock.patch('oslo_utils.uuidutils.generate_uuid')
def test_shorten_id_with_invalid_uuid(self, mock_gen_uuid):
    invalid_id = 'invalid'
    mock_gen_uuid.return_value = '1c089ea8-28fe-4f3d-8c00-f6daa2bc32f1'
    result = utils.shorten_id(invalid_id)
    expected = 10088334584203457265
    self.assertEqual(expected, result)
import base64
import hashlib
import hmac
from unittest import mock
import uuid
from osprofiler import _utils as utils
from osprofiler.tests import test
def test_shorten_id_with_valid_uuid(self):
    valid_id = '4e3e0ec6-2938-40b1-8504-09eb1d4b0dee'
    uuid_obj = uuid.UUID(valid_id)
    with mock.patch('uuid.UUID') as mock_uuid:
        mock_uuid.return_value = uuid_obj
        result = utils.shorten_id(valid_id)
        expected = 9584796812364680686
        self.assertEqual(expected, result)
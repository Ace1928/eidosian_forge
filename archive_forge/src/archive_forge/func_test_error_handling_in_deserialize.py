import datetime
from unittest import mock
import uuid
from keystone.common.cache import _context_cache
from keystone.common import utils as ks_utils
from keystone import exception
from keystone.models import receipt_model
from keystone.tests.unit import base_classes
@mock.patch.object(receipt_model.ReceiptModel, '__init__', side_effect=Exception)
def test_error_handling_in_deserialize(self, handler_mock):
    serialized = self.receipt_handler.serialize(self.exp_receipt)
    self.assertRaises(exception.CacheDeserializationError, self.receipt_handler.deserialize, serialized)
from tests.unit import unittest
from boto.dynamodb.batch import Batch
from boto.dynamodb.table import Table
from boto.dynamodb.layer2 import Layer2
from boto.dynamodb.batch import BatchList
def test_batch_to_dict(self):
    b = Batch(self.table, ['k1', 'k2'], attributes_to_get=['foo'], consistent_read=True)
    self.assertDictEqual(b.to_dict(), {'AttributesToGet': ['foo'], 'Keys': [{'HashKeyElement': {'S': 'k1'}}, {'HashKeyElement': {'S': 'k2'}}], 'ConsistentRead': True})
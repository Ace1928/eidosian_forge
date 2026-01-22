from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def dynamize_attribute_updates(self, pending_updates):
    """
        Convert a set of pending item updates into the structure
        required by Layer1.
        """
    d = {}
    for attr_name in pending_updates:
        action, value = pending_updates[attr_name]
        if value is None:
            d[attr_name] = {'Action': action}
        else:
            d[attr_name] = {'Action': action, 'Value': self.dynamizer.encode(value)}
    return d
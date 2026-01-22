from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def dynamize_expected_value(self, expected_value):
    """
        Convert an expected_value parameter into the data structure
        required for Layer1.
        """
    d = None
    if expected_value:
        d = {}
        for attr_name in expected_value:
            attr_value = expected_value[attr_name]
            if attr_value is True:
                attr_value = {'Exists': True}
            elif attr_value is False:
                attr_value = {'Exists': False}
            else:
                val = self.dynamizer.encode(expected_value[attr_name])
                attr_value = {'Value': val}
            d[attr_name] = attr_value
    return d
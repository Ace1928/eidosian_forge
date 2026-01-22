from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def dynamize_range_key_condition(self, range_key_condition):
    """
        Convert a layer2 range_key_condition parameter into the
        structure required by Layer1.
        """
    return range_key_condition.to_dict()
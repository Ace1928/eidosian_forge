from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
@property
def scanned_count(self):
    """
        As above, but representing the total number of items scanned by
        DynamoDB, without regard to any filters.
        """
    self.response
    return self._scanned_count
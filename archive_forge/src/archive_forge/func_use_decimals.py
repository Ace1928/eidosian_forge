from boto.dynamodb.layer1 import Layer1
from boto.dynamodb.table import Table
from boto.dynamodb.schema import Schema
from boto.dynamodb.item import Item
from boto.dynamodb.batch import BatchList, BatchWriteList
from boto.dynamodb.types import get_dynamodb_type, Dynamizer, \
def use_decimals(self, use_boolean=False):
    """
        Use the ``decimal.Decimal`` type for encoding/decoding numeric types.

        By default, ints/floats are used to represent numeric types
        ('N', 'NS') received from DynamoDB.  Using the ``Decimal``
        type is recommended to prevent loss of precision.

        """
    self.dynamizer = Dynamizer() if use_boolean else NonBooleanDynamizer()
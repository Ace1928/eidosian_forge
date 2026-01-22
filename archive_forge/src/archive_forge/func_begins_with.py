from collections import namedtuple
import re
from boto3.exceptions import DynamoDBOperationNotSupportedError
from boto3.exceptions import DynamoDBNeedsConditionError
from boto3.exceptions import DynamoDBNeedsKeyConditionError
def begins_with(self, value):
    """Creates a condition where the attribute begins with the value.

        :param value: The value that the attribute begins with.
        """
    return BeginsWith(self, value)
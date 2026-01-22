from boto.dynamodb.exceptions import DynamoDBItemError
@property
def range_key_name(self):
    return self._range_key_name
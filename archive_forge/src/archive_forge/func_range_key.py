from boto.dynamodb.exceptions import DynamoDBItemError
@property
def range_key(self):
    return self.get(self._range_key_name)
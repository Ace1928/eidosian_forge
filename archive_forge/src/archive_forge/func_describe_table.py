import time
from binascii import crc32
import boto
from boto.connection import AWSAuthConnection
from boto.exception import DynamoDBResponseError
from boto.provider import Provider
from boto.dynamodb import exceptions as dynamodb_exceptions
from boto.compat import json
def describe_table(self, table_name):
    """
        Returns information about the table including current
        state of the table, primary key schema and when the
        table was created.

        :type table_name: str
        :param table_name: The name of the table to describe.
        """
    data = {'TableName': table_name}
    json_input = json.dumps(data)
    return self.make_request('DescribeTable', json_input)
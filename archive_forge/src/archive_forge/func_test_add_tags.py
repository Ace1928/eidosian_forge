from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.connection import EC2Connection
from boto.ec2.ec2object import TaggedEC2Object
def test_add_tags(self):
    self.set_http_response(status_code=200)
    taggedEC2Object = TaggedEC2Object(self.service_connection)
    taggedEC2Object.id = 'i-abcd1234'
    taggedEC2Object.tags['already_present_key'] = 'already_present_value'
    taggedEC2Object.add_tags({'key1': 'value1', 'key2': 'value2'})
    self.assert_request_parameters({'ResourceId.1': 'i-abcd1234', 'Action': 'CreateTags', 'Tag.1.Key': 'key1', 'Tag.1.Value': 'value1', 'Tag.2.Key': 'key2', 'Tag.2.Value': 'value2'}, ignore_params_values=['AWSAccessKeyId', 'SignatureMethod', 'SignatureVersion', 'Timestamp', 'Version'])
    self.assertEqual(taggedEC2Object.tags, {'already_present_key': 'already_present_value', 'key1': 'value1', 'key2': 'value2'})
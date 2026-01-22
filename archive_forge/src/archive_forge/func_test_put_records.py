from boto.compat import json
from boto.kinesis.layer1 import KinesisConnection
from tests.unit import AWSMockServiceTestCase
def test_put_records(self):
    self.set_http_response(status_code=200)
    record_binary = {'Data': b'\x00\x01\x02\x03\x04\x05', 'PartitionKey': 'partition-key'}
    record_str = {'Data': 'data', 'PartitionKey': 'partition-key'}
    self.service_connection.put_records(stream_name='stream-name', records=[record_binary, record_str])
    body = json.loads(self.actual_request.body.decode('utf-8'))
    self.assertEqual(body['Records'][0]['Data'], 'AAECAwQF')
    self.assertEqual(body['Records'][1]['Data'], 'ZGF0YQ==')
    target = self.actual_request.headers['X-Amz-Target']
    self.assertTrue('PutRecord' in target)
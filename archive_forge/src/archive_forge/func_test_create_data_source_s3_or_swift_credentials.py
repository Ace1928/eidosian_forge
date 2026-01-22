from saharaclient.api import data_sources as ds
from saharaclient.tests.unit import base
from unittest import mock
from oslo_serialization import jsonutils as json
@mock.patch('saharaclient.api.base.ResourceManager._create')
def test_create_data_source_s3_or_swift_credentials(self, create):
    self.client.data_sources.create('ds', '', 'swift', 'swift://path')
    self.assertNotIn('credentials', create.call_args[0][1])
    self.client.data_sources.create('ds', '', 'swift', 'swift://path', credential_user='user')
    self.assertIn('credentials', create.call_args[0][1])
    self.client.data_sources.create('ds', '', 'swift', 'swift://path', s3_credentials={'accesskey': 'a'})
    self.assertIn('credentials', create.call_args[0][1])
    self.assertIn('accesskey', create.call_args[0][1]['credentials'])
    self.client.data_sources.create('ds', '', 's3', 's3://path', credential_user='swift_user', s3_credentials={'accesskey': 's3_a'})
    self.assertIn('user', create.call_args[0][1]['credentials'])
    self.assertNotIn('accesskey', create.call_args[0][1]['credentials'])
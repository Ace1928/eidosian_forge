from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_env_profile_loads_profile(self):
    self.environ['AWS_PROFILE'] = 'foo'
    self.shared_config = {'default': {'aws_access_key_id': 'shared_access_key', 'aws_secret_access_key': 'shared_secret_key'}, 'foo': {'aws_access_key_id': 'shared_access_key_foo', 'aws_secret_access_key': 'shared_secret_key_foo'}}
    self.config = {'profile foo': {'aws_access_key_id': 'cfg_access_key_foo', 'aws_secret_access_key': 'cfg_secret_key_foo'}, 'Credentials': {'aws_access_key_id': 'cfg_access_key', 'aws_secret_access_key': 'cfg_secret_key'}}
    p = provider.Provider('aws')
    self.assertEqual(p.access_key, 'shared_access_key_foo')
    self.assertEqual(p.secret_key, 'shared_secret_key_foo')
    self.assertIsNone(p.security_token)
    self.shared_config = {}
    p = provider.Provider('aws')
    self.assertEqual(p.access_key, 'cfg_access_key_foo')
    self.assertEqual(p.secret_key, 'cfg_secret_key_foo')
    self.assertIsNone(p.security_token)
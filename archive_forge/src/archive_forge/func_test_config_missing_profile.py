from datetime import datetime, timedelta
from tests.compat import mock, unittest
import os
from boto import provider
from boto.compat import expanduser
from boto.exception import InvalidInstanceMetadataError
def test_config_missing_profile(self):
    self.shared_config = {'default': {'aws_access_key_id': 'shared_access_key', 'aws_secret_access_key': 'shared_secret_key'}}
    self.config = {'Credentials': {'aws_access_key_id': 'default_access_key', 'aws_secret_access_key': 'default_secret_key'}}
    with self.assertRaises(provider.ProfileNotFoundError):
        provider.Provider('aws', profile_name='doesntexist')
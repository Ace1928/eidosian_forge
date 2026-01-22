import copy
from unittest import mock
from keystoneauth1 import session
from oslo_utils import uuidutils
import novaclient.api_versions
import novaclient.client
import novaclient.extension
from novaclient.tests.unit import utils
import novaclient.v2.client
@mock.patch('novaclient.client.warnings')
def test__check_arguments(self, mock_warnings):
    release = 'Coolest'
    novaclient.client._check_arguments({}, release=release, deprecated_name='foo')
    self.assertFalse(mock_warnings.warn.called)
    novaclient.client._check_arguments({}, release=release, deprecated_name='foo', right_name='bar')
    self.assertFalse(mock_warnings.warn.called)
    original_kwargs = {'foo': 'text'}
    actual_kwargs = copy.copy(original_kwargs)
    self.assertEqual(original_kwargs, actual_kwargs)
    novaclient.client._check_arguments(actual_kwargs, release=release, deprecated_name='foo', right_name='bar')
    self.assertNotEqual(original_kwargs, actual_kwargs)
    self.assertEqual({'bar': original_kwargs['foo']}, actual_kwargs)
    self.assertTrue(mock_warnings.warn.called)
    mock_warnings.warn.reset_mock()
    original_kwargs = {'foo': 'text'}
    actual_kwargs = copy.copy(original_kwargs)
    self.assertEqual(original_kwargs, actual_kwargs)
    novaclient.client._check_arguments(actual_kwargs, release=release, deprecated_name='foo')
    self.assertNotEqual(original_kwargs, actual_kwargs)
    self.assertEqual({}, actual_kwargs)
    self.assertTrue(mock_warnings.warn.called)
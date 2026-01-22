import json
import tempfile
from unittest import mock
import io
from oslo_serialization import base64
import testtools
from testtools import matchers
from urllib import error
import yaml
from heatclient.common import template_utils
from heatclient.common import utils
from heatclient import exc
@mock.patch('urllib.request.urlopen')
def test_process_multiple_environments_and_files_tracker(self, mock_url):
    env_file1 = '/home/my/dir/env1.yaml'
    env1 = b'\n        parameters:\n          "param1": "value1"\n        resource_registry:\n          "OS::Thingy1": "file:///home/b/a.yaml"\n        '
    mock_url.side_effect = [io.BytesIO(env1), io.BytesIO(self.template_a), io.BytesIO(self.template_a)]
    env_file_list = []
    files, env = template_utils.process_multiple_environments_and_files([env_file1], env_list_tracker=env_file_list)
    expected_env = {'parameters': {'param1': 'value1'}, 'resource_registry': {'OS::Thingy1': 'file:///home/b/a.yaml'}}
    self.assertEqual(expected_env, env)
    self.assertEqual(self.template_a.decode('utf-8'), files['file:///home/b/a.yaml'])
    self.assertEqual(['file:///home/my/dir/env1.yaml'], env_file_list)
    self.assertIn('file:///home/my/dir/env1.yaml', files)
    self.assertEqual(expected_env, json.loads(files['file:///home/my/dir/env1.yaml']))
    mock_url.assert_has_calls([mock.call('file://%s' % env_file1), mock.call('file:///home/b/a.yaml'), mock.call('file:///home/b/a.yaml')])
import os
import unittest
import mock
from .config_exception import ConfigException
from .exec_provider import ExecProvider
from .kube_config import ConfigNode
@mock.patch('subprocess.Popen')
def test_nonjson_output_returned(self, mock):
    instance = mock.return_value
    instance.wait.return_value = 0
    instance.communicate.return_value = ('', '')
    with self.assertRaises(ConfigException) as context:
        ep = ExecProvider(self.input_ok)
        ep.run()
    self.assertIn('exec: failed to decode process output', context.exception.args[0])
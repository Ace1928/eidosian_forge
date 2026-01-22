import os
import unittest
import mock
from .config_exception import ConfigException
from .exec_provider import ExecProvider
from .kube_config import ConfigNode
@mock.patch('subprocess.Popen')
def test_ok_01(self, mock):
    instance = mock.return_value
    instance.wait.return_value = 0
    instance.communicate.return_value = (self.output_ok, '')
    ep = ExecProvider(self.input_ok)
    result = ep.run()
    self.assertTrue(isinstance(result, dict))
    self.assertTrue('token' in result)
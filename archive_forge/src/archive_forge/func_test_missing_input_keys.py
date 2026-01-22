import os
import unittest
import mock
from .config_exception import ConfigException
from .exec_provider import ExecProvider
from .kube_config import ConfigNode
def test_missing_input_keys(self):
    exec_configs = [ConfigNode('test1', {}), ConfigNode('test2', {'command': ''}), ConfigNode('test3', {'apiVersion': ''})]
    for exec_config in exec_configs:
        with self.assertRaises(ConfigException) as context:
            ExecProvider(exec_config)
        self.assertIn('exec: malformed request. missing key', context.exception.args[0])
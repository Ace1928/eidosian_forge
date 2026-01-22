import os
import unittest
import mock
from .config_exception import ConfigException
from .exec_provider import ExecProvider
from .kube_config import ConfigNode
@mock.patch('subprocess.Popen')
def test_mismatched_api_version(self, mock):
    instance = mock.return_value
    instance.wait.return_value = 0
    wrong_api_version = 'client.authentication.k8s.io/v1'
    output = '\n        {\n            "apiVersion": "%s",\n            "kind": "ExecCredential",\n            "status": {\n                "token": "dummy"\n            }\n        }\n        ' % wrong_api_version
    instance.communicate.return_value = (output, '')
    with self.assertRaises(ConfigException) as context:
        ep = ExecProvider(self.input_ok)
        ep.run()
    self.assertIn('exec: plugin api version %s does not match' % wrong_api_version, context.exception.args[0])
import base64
import datetime
import json
import os
import shutil
import tempfile
import unittest
import mock
from ruamel import yaml
from six import PY3, next
from kubernetes.client import Configuration
from .config_exception import ConfigException
from .kube_config import (ENV_KUBECONFIG_PATH_SEPARATOR, ConfigNode, FileOrData,
@mock.patch('kubernetes.config.kube_config.ExecProvider.run')
def test_user_exec_auth(self, mock):
    token = 'dummy'
    mock.return_value = {'token': token}
    expected = FakeConfig(host=TEST_HOST, api_key={'authorization': BEARER_TOKEN_FORMAT % token})
    actual = FakeConfig()
    KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='exec_cred_user').load_and_set(actual)
    self.assertEqual(expected, actual)
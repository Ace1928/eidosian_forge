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
def test_no_users_section(self):
    expected = FakeConfig(host=TEST_HOST)
    actual = FakeConfig()
    test_kube_config = self.TEST_KUBE_CONFIG.copy()
    del test_kube_config['users']
    KubeConfigLoader(config_dict=test_kube_config, active_context='gcp').load_and_set(actual)
    self.assertEqual(expected, actual)
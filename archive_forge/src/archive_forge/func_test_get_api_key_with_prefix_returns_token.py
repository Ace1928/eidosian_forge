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
def test_get_api_key_with_prefix_returns_token(self):
    expected_token = 'expected_token'
    config = Configuration()
    config.api_key['authorization'] = expected_token
    self.assertEqual(expected_token, config.get_api_key_with_prefix('authorization'))
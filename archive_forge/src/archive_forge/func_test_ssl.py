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
def test_ssl(self):
    expected = FakeConfig(host=TEST_SSL_HOST, token=BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, cert_file=self._create_temp_file(TEST_CLIENT_CERT), key_file=self._create_temp_file(TEST_CLIENT_KEY), ssl_ca_cert=self._create_temp_file(TEST_CERTIFICATE_AUTH))
    actual = FakeConfig()
    KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='ssl').load_and_set(actual)
    self.assertEqual(expected, actual)
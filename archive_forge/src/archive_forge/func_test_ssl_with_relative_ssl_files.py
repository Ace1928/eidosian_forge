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
def test_ssl_with_relative_ssl_files(self):
    expected = FakeConfig(host=TEST_SSL_HOST, token=BEARER_TOKEN_FORMAT % TEST_DATA_BASE64, cert_file=self._create_temp_file(TEST_CLIENT_CERT), key_file=self._create_temp_file(TEST_CLIENT_KEY), ssl_ca_cert=self._create_temp_file(TEST_CERTIFICATE_AUTH))
    try:
        temp_dir = tempfile.mkdtemp()
        actual = FakeConfig()
        with open(os.path.join(temp_dir, 'cert_test'), 'wb') as fd:
            fd.write(TEST_CERTIFICATE_AUTH.encode())
        with open(os.path.join(temp_dir, 'client_cert'), 'wb') as fd:
            fd.write(TEST_CLIENT_CERT.encode())
        with open(os.path.join(temp_dir, 'client_key'), 'wb') as fd:
            fd.write(TEST_CLIENT_KEY.encode())
        with open(os.path.join(temp_dir, 'token_file'), 'wb') as fd:
            fd.write(TEST_DATA_BASE64.encode())
        KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='ssl-local-file', config_base_path=temp_dir).load_and_set(actual)
        self.assertEqual(expected, actual)
    finally:
        shutil.rmtree(temp_dir)
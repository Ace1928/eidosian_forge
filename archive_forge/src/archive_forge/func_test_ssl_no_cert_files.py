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
def test_ssl_no_cert_files(self):
    loader = KubeConfigLoader(config_dict=self.TEST_KUBE_CONFIG, active_context='ssl-no_file')
    self.expect_exception(loader.load_and_set, 'does not exists', FakeConfig())
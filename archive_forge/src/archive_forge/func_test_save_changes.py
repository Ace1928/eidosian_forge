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
def test_save_changes(self):
    kubeconfigs = self._create_multi_config()
    kconf = KubeConfigMerger(kubeconfigs)
    user = kconf.config['users'].get_with_name('expired_oidc')['user']
    provider = user['auth-provider']['config']
    provider.value['id-token'] = 'token-changed'
    kconf.save_changes()
    kconf = KubeConfigMerger(kubeconfigs)
    user = kconf.config['users'].get_with_name('expired_oidc')['user']
    provider = user['auth-provider']['config']
    self.assertEqual(provider.value['id-token'], 'token-changed')
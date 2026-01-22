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
def test_list_kube_config_contexts(self):
    kubeconfigs = self._create_multi_config()
    expected_contexts = [{'context': {'cluster': 'default'}, 'name': 'no_user'}, {'context': {'cluster': 'ssl', 'user': 'ssl'}, 'name': 'ssl'}, {'context': {'cluster': 'default', 'user': 'simple_token'}, 'name': 'simple_token'}, {'context': {'cluster': 'default', 'user': 'expired_oidc'}, 'name': 'expired_oidc'}]
    contexts, active_context = list_kube_config_contexts(config_file=kubeconfigs)
    self.assertEqual(contexts, expected_contexts)
    self.assertEqual(active_context, expected_contexts[0])
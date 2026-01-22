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
def test_normal_map_array_operations(self):
    self.assertEqual('test', self.node['key1'])
    self.assertEqual(5, len(self.node))
    self.assertEqual('test_obj/key2', self.node['key2'].name)
    self.assertEqual(['a', 'b', 'c'], self.node['key2'].value)
    self.assertEqual('b', self.node['key2'][1])
    self.assertEqual(3, len(self.node['key2']))
    self.assertEqual('test_obj/key3', self.node['key3'].name)
    self.assertEqual({'inner_key': 'inner_value'}, self.node['key3'].value)
    self.assertEqual('inner_value', self.node['key3']['inner_key'])
    self.assertEqual(1, len(self.node['key3']))
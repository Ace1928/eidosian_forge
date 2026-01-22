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
def expect_exception(self, func, message_part, *args, **kwargs):
    with self.assertRaises(ConfigException) as context:
        func(*args, **kwargs)
    self.assertIn(message_part, str(context.exception))
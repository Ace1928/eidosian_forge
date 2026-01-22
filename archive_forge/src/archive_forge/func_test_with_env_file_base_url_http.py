import json
import tempfile
from unittest import mock
import io
from oslo_serialization import base64
import testtools
from testtools import matchers
from urllib import error
import yaml
from heatclient.common import template_utils
from heatclient.common import utils
from heatclient import exc
def test_with_env_file_base_url_http(self):
    url = 'http://no.where/path/to/a.yaml'
    env = '\n        resource_registry:\n          resources:\n            server_for_me:\n              "OS::Thingy": to/a.yaml\n        '
    env_base_url = 'http://no.where/path'
    self.collect_links(env, self.template_a, url, env_base_url)
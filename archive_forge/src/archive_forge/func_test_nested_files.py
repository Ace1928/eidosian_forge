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
def test_nested_files(self):
    url = 'file:///home/b/a.yaml'
    env = '\n        resource_registry:\n          resources:\n            freddy:\n              "OS::Thingy": "%s"\n        ' % url
    self.collect_links(env, self.template_a, url)
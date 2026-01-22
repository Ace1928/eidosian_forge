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
def test_with_built_in_provider(self):
    env = '\n        resource_registry:\n          resources:\n            server_for_me:\n              "OS::Thingy": OS::Compute::Server\n        '
    self.collect_links(env, self.template_a, None)
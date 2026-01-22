from unittest import mock
import testtools
from urllib import parse
from heatclient.common import utils
from heatclient.v1 import resources
def test_stack_name_no_links(self):
    resource = resources.Resource(None, {})
    self.assertIsNone(resource.stack_name)
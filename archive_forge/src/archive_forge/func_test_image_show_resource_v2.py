from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_show_resource_v2(self):
    self.my_image.resource_id = 'test_image_id'
    image = {'key1': 'val1', 'key2': 'val2'}
    self.images.get.return_value = image
    self.glanceclient.version = 2.0
    self.assertEqual({'key1': 'val1', 'key2': 'val2'}, self.my_image.FnGetAtt('show'))
    self.images.get.assert_called_once_with('test_image_id')
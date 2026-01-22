from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_active_property_image_status_killed(self):
    self.images.reactivate.return_value = None
    self.images.deactivate.return_value = None
    value = mock.MagicMock()
    image_id = '41f0e60c-ebb4-4375-a2b4-845ae8b9c995'
    value.id = image_id
    value.status = 'killed'
    self.my_image.resource_id = image_id
    self.images.create.return_value = value
    self.images.get.return_value = value
    ex = self.assertRaises(exception.ResourceInError, self.my_image.check_create_complete, False)
    self.assertIn('killed', ex.message)
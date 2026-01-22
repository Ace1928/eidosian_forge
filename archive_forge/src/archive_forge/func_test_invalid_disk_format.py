from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_invalid_disk_format(self):
    tpl = template_format.parse(image_download_template_validate)
    stack = parser.Stack(self.ctx, 'glance_image_stack_validate', template.Template(tpl))
    image = stack['image']
    props = stack.t.t['resources']['image']['properties'].copy()
    props['disk_format'] = 'incorrect_format'
    image.t = image.t.freeze(properties=props)
    image.reparse()
    error_msg = 'Property error: resources.image.properties.disk_format: "incorrect_format" is not an allowed value'
    self._test_validate(image, error_msg)
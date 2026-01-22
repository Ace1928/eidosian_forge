from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_invalid_min_ram(self):
    tpl = template_format.parse(image_download_template_validate)
    stack = parser.Stack(self.ctx, 'glance_image_stack_validate', template.Template(tpl))
    image = stack['image']
    props = stack.t.t['resources']['image']['properties'].copy()
    props['min_ram'] = -1
    image.t = image.t.freeze(properties=props)
    image.reparse()
    error_msg = 'Property error: resources.image.properties.min_ram: -1 is out of range (min: 0, max: None)'
    self._test_validate(image, error_msg)
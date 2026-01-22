from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_miss_location(self):
    tpl = template_format.parse(image_download_template_validate)
    stack = parser.Stack(self.ctx, 'glance_image_stack_validate', template.Template(tpl))
    image = stack['image']
    props = stack.t.t['resources']['image']['properties'].copy()
    del props['location']
    image.t = image.t.freeze(properties=props)
    image.reparse()
    error_msg = 'Property location not assigned'
    self._test_validate(image, error_msg)
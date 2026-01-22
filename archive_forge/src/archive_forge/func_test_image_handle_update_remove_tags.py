from unittest import mock
from glanceclient import exc
from heat.common import exception
from heat.common import template_format
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_image_handle_update_remove_tags(self):
    self.my_image.resource_id = '477e8273-60a7-4c41-b683-fdb0bc7cd151'
    props = self.stack.t.t['resources']['my_image']['properties'].copy()
    props['tags'] = ['tag1']
    self.my_image.t = self.my_image.t.freeze(properties=props)
    self.my_image.reparse()
    prop_diff = {'tags': None}
    self.my_image.handle_update(json_snippet=None, tmpl_diff=None, prop_diff=prop_diff)
    self.image_tags.delete.assert_called_once_with(self.my_image.resource_id, 'tag1')
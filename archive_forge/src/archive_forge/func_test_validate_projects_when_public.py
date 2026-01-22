import collections
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import cinder as c_plugin
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_validate_projects_when_public(self):
    tmpl = self.stack.t.t
    props = tmpl['resources']['my_volume_type']['properties'].copy()
    props['is_public'] = True
    props['projects'] = ['id1']
    self.my_volume_type.t = self.my_volume_type.t.freeze(properties=props)
    self.my_volume_type.reparse()
    self.cinderclient.volume_api_version = 3
    self.stub_KeystoneProjectConstraint()
    ex = self.assertRaises(exception.StackValidationFailed, self.my_volume_type.validate)
    expected = 'Can not specify property "projects" if the volume type is public.'
    self.assertEqual(expected, str(ex))
from unittest import mock
from heat.common import exception
from heat.engine import environment
from heat.engine import resource as res
from heat.engine import service
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_resource_schema_for_hidden_type(self):
    type_name = 'ResourceTypeHidden'
    self.assertRaises(exception.NotSupported, self.eng.resource_schema, self.ctx, type_name)
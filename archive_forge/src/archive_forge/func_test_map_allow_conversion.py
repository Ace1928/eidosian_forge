from unittest import mock
from oslo_serialization import jsonutils
from heat.common import exception
from heat.engine import constraints
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters as hot_param
from heat.engine import parameters
from heat.engine import plugin_manager
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import support
from heat.engine import translation
from heat.tests import common
def test_map_allow_conversion(self):
    p = properties.Property({'Type': 'Map'})
    p.schema.allow_conversion = True
    self.assertEqual('foo', p.get_value('foo'))
    self.assertEqual(jsonutils.dumps(['foo']), p.get_value(['foo']))
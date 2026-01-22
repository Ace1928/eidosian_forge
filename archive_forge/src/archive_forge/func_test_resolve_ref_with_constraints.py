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
def test_resolve_ref_with_constraints(self):

    class IncorrectConstraint(constraints.BaseCustomConstraint):
        expected_exceptions = (Exception,)

        def validate_with_client(self, client, value):
            raise Exception('Test exception')

    class TestCustomConstraint(constraints.CustomConstraint):

        @property
        def custom_constraint(self):
            return IncorrectConstraint()
    schema = {'foo': properties.Schema(properties.Schema.STRING, constraints=[TestCustomConstraint('test_constraint')])}

    def test_resolver(prop, nullable=False):
        return 'None'

    class rsrc(object):
        action = INIT = 'INIT'

    class DummyStack(dict):
        pass
    stack = DummyStack(another_res=rsrc())
    props = properties.Properties(schema, {'foo': hot_funcs.GetResource(stack, 'get_resource', 'another_res')}, test_resolver)
    try:
        self.assertIsNone(props.validate())
    except exception.StackValidationFailed:
        self.fail('Constraints should not have been evaluated.')
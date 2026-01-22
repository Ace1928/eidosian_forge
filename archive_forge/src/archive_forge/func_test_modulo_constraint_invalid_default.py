import copy
from unittest import mock
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine.cfn import functions as cfn_functions
from heat.engine.cfn import parameters as cfn_param
from heat.engine import conditions
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import functions as hot_functions
from heat.engine.hot import parameters as hot_param
from heat.engine.hot import template as hot_template
from heat.engine import resource
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_modulo_constraint_invalid_default(self):
    modulo_desc = 'Value must be an odd number'
    modulo_name = 'ControllerCount'
    param = {modulo_name: {'description': 'Number of controller nodes', 'type': 'number', 'default': 2, 'constraints': [{'modulo': {'step': 2, 'offset': 1}, 'description': modulo_desc}]}}
    schema = hot_param.HOTParamSchema20170224.from_dict(modulo_name, param[modulo_name])
    err = self.assertRaises(exception.InvalidSchemaError, schema.validate)
    self.assertIn(modulo_desc, str(err))
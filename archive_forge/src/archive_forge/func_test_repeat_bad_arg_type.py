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
def test_repeat_bad_arg_type(self):
    tmpl = template.Template(hot_kilo_tpl_empty)
    snippet = {'repeat': {'template': 'this is %var%', 'for_each': '%var%'}}
    repeat = tmpl.parse(None, snippet)
    regxp = 'repeat: The "for_each" argument to "repeat" must contain a map'
    self.assertRaisesRegex(exception.StackValidationFailed, regxp, function.validate, repeat)
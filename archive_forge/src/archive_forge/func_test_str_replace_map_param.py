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
def test_str_replace_map_param(self):
    """Test old str_replace function with non-string map param."""
    snippet = {'str_replace': {'template': 'jsonvar1', 'params': {'jsonvar1': {'foo': 123}}}}
    tmpl = template.Template(hot_tpl_empty)
    ex = self.assertRaises(TypeError, self.resolve, snippet, tmpl)
    self.assertIn('"str_replace" params must be strings or numbers, param jsonvar1 is not valid', str(ex))
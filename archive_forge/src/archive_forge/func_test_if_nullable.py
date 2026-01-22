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
def test_if_nullable(self):
    snippet = {'single': {'if': [False, 'value_if_true']}, 'nested_true': {'if': [True, {'if': [False, 'foo']}, 'bar']}, 'nested_false': {'if': [False, 'baz', {'if': [False, 'quux']}]}, 'control': {'if': [False, True, None]}}
    tmpl = template.Template(hot_wallaby_tpl_empty)
    resolved = self.resolve(snippet, tmpl, None)
    self.assertEqual({'control': None}, resolved)
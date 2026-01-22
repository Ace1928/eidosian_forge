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
def test_str_replace_strict_invalid_param_keys(self):
    """Test str_replace function parameter keys.

        Pass wrong parameters to function and verify that we get
        a KeyError.
        """
    snippets = [{'str_replace_strict': {'t': 'Template var1 string var2', 'params': {'var1': 'foo', 'var2': 'bar'}}}, {'str_replace_strict': {'template': 'Template var1 string var2', 'param': {'var1': 'foo', 'var2': 'bar'}}}]
    for snippet in snippets:
        tmpl = template.Template(hot_ocata_tpl_empty)
        ex = self.assertRaises(exception.StackValidationFailed, self.resolve, snippet, tmpl)
    self.assertIn('"str_replace_strict" syntax should be str_replace_strict:\\n', str(ex))
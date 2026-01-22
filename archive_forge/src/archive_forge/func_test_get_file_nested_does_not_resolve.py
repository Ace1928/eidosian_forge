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
def test_get_file_nested_does_not_resolve(self):
    """Test get_file function does not resolve nested calls."""
    snippet = {'get_file': 'file:///tmp/foo.yaml'}
    snippet_resolved = '{get_file: file:///tmp/bar.yaml}'
    tmpl = template.Template(hot_tpl_empty, files={'file:///tmp/foo.yaml': snippet_resolved, 'file:///tmp/bar.yaml': 'bar content'})
    stack = parser.Stack(utils.dummy_context(), 'param_id_test', tmpl)
    self.assertEqual(snippet_resolved, self.resolve(snippet, tmpl, stack))
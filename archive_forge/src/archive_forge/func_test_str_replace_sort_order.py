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
def test_str_replace_sort_order(self):
    """Test str_replace function replacement order."""
    snippet = {'str_replace': {'template': '9876543210', 'params': {'987654': 'a', '876543': 'b', '765432': 'c', '654321': 'd', '543210': 'e'}}}
    tmpl = template.Template(hot_tpl_empty)
    self.assertEqual('9876e', self.resolve(snippet, tmpl))
import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_invalid_version_not_in_heat_versions(self):
    invalid_heat_version_tmp = template_format.parse('{\n            "HeatTemplateFormatVersion" : "2010-09-09",\n            }')
    versions = {('HeatTemplateFormatVersion', '2012-12-12'): cfn_t.CfnTemplate, ('HeatTemplateFormatVersion', '2014-12-12'): cfn_t.CfnTemplate}
    temp_copy = copy.deepcopy(template._template_classes)
    template._template_classes = versions
    init_ex = self.assertRaises(exception.InvalidTemplateVersion, template.Template, invalid_heat_version_tmp)
    ex_error_msg = 'The template version is invalid: "HeatTemplateFormatVersion: 2010-09-09". "HeatTemplateFormatVersion" should be one of: 2012-12-12, 2014-12-12'
    self.assertEqual(ex_error_msg, str(init_ex))
    template._template_classes = temp_copy
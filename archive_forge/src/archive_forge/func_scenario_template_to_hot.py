from heat.db import api as db_api
from heat.engine import service
from heat.engine import stack
from heat.tests.convergence.framework import message_processor
from heat.tests.convergence.framework import message_queue
from heat.tests.convergence.framework import scenario_template
from heat.tests import utils
def scenario_template_to_hot(self, scenario_tmpl):
    """Converts the scenario template into hot template."""
    hot_tmpl = {'heat_template_version': '2013-05-23'}
    resources = {}
    for res_name, res_def in scenario_tmpl.resources.items():
        props = getattr(res_def, 'properties')
        depends = getattr(res_def, 'depends_on')
        res_defn = {'type': 'OS::Heat::TestResource'}
        if props:
            props_def = {}
            for prop_name, prop_value in props.items():
                if isinstance(prop_value, scenario_template.GetAtt):
                    prop_res = getattr(prop_value, 'target_name')
                    prop_attr = getattr(prop_value, 'attr')
                    prop_value = {'get_attr': [prop_res, prop_attr]}
                elif isinstance(prop_value, scenario_template.GetRes):
                    prop_res = getattr(prop_value, 'target_name')
                    prop_value = {'get_resource': prop_res}
                props_def[prop_name] = prop_value
            res_defn['properties'] = props_def
        if depends:
            res_defn['depends_on'] = depends
        resources[res_name] = res_defn
    hot_tmpl['resources'] = resources
    return hot_tmpl
from .. import check, controldir, errors, tests
from ..upgrade import upgrade
from .scenarios import load_tests_apply_scenarios
def upgrade_scenarios():
    scenario_pairs = [('knit', '1.6', False), ('1.6', '1.6.1-rich-root', True)]
    scenarios = []
    for old_name, new_name, model_change in scenario_pairs:
        name = old_name + ', ' + new_name
        scenarios.append((name, dict(scenario_old_format=old_name, scenario_new_format=new_name, scenario_model_change=model_change)))
    return scenarios
import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_orchestration_template_validate_yaml(self):
    self._orchestration_template_validate('heat_minimal.yaml', ['ClientName=ClientName', 'WaitSecs=123'])
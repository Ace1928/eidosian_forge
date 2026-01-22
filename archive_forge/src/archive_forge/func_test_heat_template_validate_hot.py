import yaml
from tempest.lib import exceptions
from heatclient.tests.functional.osc.v1 import base
def test_heat_template_validate_hot(self):
    self._template_validate('heat_minimal_hot.yaml', ['test_client_name=test_client_name', 'test_wait_secs=123'])
import subprocess
from oslotest import base
def test_parser_options(self):
    output = subprocess.check_output(['openstack', '--help'])
    self.assertIn('--os-placement-api-version', output.decode('utf-8'))
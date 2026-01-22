from unittest import mock
import yaml
from heat.common import environment_format
from heat.tests import common
def test_wrong_sections(self):
    env = '\nparameters: {}\nresource_regis: {}\n'
    self.assertRaises(ValueError, environment_format.parse, env)
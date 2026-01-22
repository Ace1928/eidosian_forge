import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
def test_heat_help_cmd(self):
    help_text = self.heat('help resource-template')
    lines = help_text.split('\n')
    self.assertFirstLineStartsWith(lines, 'usage: heat resource-template')
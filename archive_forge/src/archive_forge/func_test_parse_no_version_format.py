import os
from unittest import mock
import re
import yaml
from heat.common import config
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.tests import common
from heat.tests import utils
def test_parse_no_version_format(self):
    yaml = ''
    self._parse_template(yaml, 'Template format version not found')
    yaml2 = 'Parameters: {}\nMappings: {}\nResources: {}\nOutputs: {}\n'
    self._parse_template(yaml2, 'Template format version not found')
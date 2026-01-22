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
def test_parse_to_value_exception(self):
    text = 'not important'
    with mock.patch.object(yaml, 'load') as yaml_loader:
        yaml_loader.side_effect = self.raised_exception
        err = self.assertRaises(ValueError, template_format.parse, text, 'file://test.yaml')
        self.assertIn('Error parsing template file://test.yaml', str(err))
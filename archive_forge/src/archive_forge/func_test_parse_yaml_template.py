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
def test_parse_yaml_template(self):
    tmpl_str = 'heat_template_version: 2013-05-23'
    expected = {'heat_template_version': '2013-05-23'}
    self.assertEqual(expected, template_format.parse(tmpl_str))
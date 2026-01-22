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
def test_parse_empty_json_template(self):
    tmpl_str = '{}'
    msg = 'Template format version not found'
    self._parse_template(tmpl_str, msg)
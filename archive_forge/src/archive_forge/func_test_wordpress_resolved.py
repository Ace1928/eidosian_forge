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
def test_wordpress_resolved(self):
    self.compare_stacks('WordPress_Single_Instance.template', 'WordPress_Single_Instance.yaml', {'KeyName': 'test'})
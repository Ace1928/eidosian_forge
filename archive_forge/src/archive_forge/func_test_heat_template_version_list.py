import json
import os
from tempest.lib import exceptions
import yaml
from heatclient.tests.functional import base
def test_heat_template_version_list(self):
    ret = self.heat('template-version-list')
    tmpl_types = self.parser.listing(ret)
    self.assertTableStruct(tmpl_types, ['version', 'type'])
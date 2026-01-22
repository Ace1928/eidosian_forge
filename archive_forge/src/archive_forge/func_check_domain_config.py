import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def check_domain_config(self, config, config_ref):
    for attr in config_ref:
        self.assertEqual(getattr(config, attr), config_ref[attr], 'Expected different %s' % attr)
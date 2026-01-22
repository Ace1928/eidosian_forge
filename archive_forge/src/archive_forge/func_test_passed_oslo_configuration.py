from unittest import mock
import uuid
from oslo_config import cfg
from oslotest import createfile
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _opts
from keystonemiddleware.tests.unit.auth_token import base
def test_passed_oslo_configuration(self):
    conf = {'oslo_config_config': self.local_oslo_config}
    app = self._create_app(conf)
    for option in self.oslo_options:
        self.assertEqual(self.oslo_options[option], conf_get(app, option))
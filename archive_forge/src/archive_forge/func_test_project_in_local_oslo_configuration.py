from unittest import mock
import uuid
from oslo_config import cfg
from oslotest import createfile
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _opts
from keystonemiddleware.tests.unit.auth_token import base
def test_project_in_local_oslo_configuration(self):
    conf = {'oslo_config_project': self.project, 'oslo_config_file': self.conf_file_fixture.path}
    app = self._create_app(conf)
    for option in self.file_options:
        self.assertEqual(self.file_options[option], conf_get(app, option), option)
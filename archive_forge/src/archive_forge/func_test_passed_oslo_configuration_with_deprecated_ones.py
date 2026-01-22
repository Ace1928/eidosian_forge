from unittest import mock
import uuid
from oslo_config import cfg
from oslotest import createfile
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _opts
from keystonemiddleware.tests.unit.auth_token import base
def test_passed_oslo_configuration_with_deprecated_ones(self):
    deprecated_opt = cfg.IntOpt('test_opt', deprecated_for_removal=True)
    cfg.CONF.register_opt(deprecated_opt)
    cfg.CONF(args=[], default_config_files=[self.conf_file_fixture.path])
    conf = {'oslo_config_config': cfg.CONF}
    self._create_app(conf)
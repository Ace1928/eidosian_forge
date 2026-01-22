import tempfile
from oslo_config import cfg
from oslo_config import fixture
from oslotest import base
from castellan import _config_driver
from castellan.common.objects import opaque_data
from castellan.tests.unit.key_manager import fake
def test_complete_driver(self):
    self.conf_fixture.load_raw_values(group='castellan_source', driver='castellan', config_file='config.conf', mapping_file='mapping.conf')
    with base.mock.patch.object(_config_driver, 'CastellanConfigurationSource') as source_class:
        self.driver.open_source_from_opt_group(self.conf, 'castellan_source')
        source_class.assert_called_once_with('castellan_source', self.conf.castellan_source.config_file, self.conf.castellan_source.mapping_file)
import tempfile
from oslo_config import cfg
from oslo_config import fixture
from oslotest import base
from castellan import _config_driver
from castellan.common.objects import opaque_data
from castellan.tests.unit.key_manager import fake
def test_fetch_secret(self):
    km = fake.fake_api()
    secret_id = km.store('fake_context', opaque_data.OpaqueData(b'super_secret!'))
    config = '[key_manager]\nbackend=vault'
    mapping = '[DEFAULT]\nmy_secret=' + secret_id
    with tempfile.NamedTemporaryFile() as config_file:
        config_file.write(config.encode('utf-8'))
        config_file.flush()
        with tempfile.NamedTemporaryFile() as mapping_file:
            mapping_file.write(mapping.encode('utf-8'))
            mapping_file.flush()
            self.conf_fixture.load_raw_values(group='castellan_source', driver='castellan', config_file=config_file.name, mapping_file=mapping_file.name)
            source = self.driver.open_source_from_opt_group(self.conf, 'castellan_source')
            source._mngr = km
            self.assertEqual('super_secret!', source.get('DEFAULT', 'my_secret', cfg.StrOpt(''))[0])
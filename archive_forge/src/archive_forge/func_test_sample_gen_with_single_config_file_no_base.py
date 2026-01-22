from unittest import mock
from oslotest import base
from oslo_config import sphinxconfiggen
@mock.patch('os.path.isdir')
@mock.patch('os.path.isfile')
@mock.patch('oslo_config.generator.main')
def test_sample_gen_with_single_config_file_no_base(self, main, isfile, isdir):
    isfile.side_effect = [False, True]
    isdir.return_value = True
    config = mock.Mock(config_generator_config_file='nova-gen.conf', sample_config_basename=None)
    app = mock.Mock(srcdir='/opt/nova', config=config)
    sphinxconfiggen.generate_sample(app)
    main.assert_called_once_with(args=['--config-file', '/opt/nova/nova-gen.conf', '--output-file', '/opt/nova/sample.config'])
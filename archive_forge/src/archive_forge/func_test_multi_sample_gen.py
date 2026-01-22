from unittest import mock
from oslotest import base
from oslo_config import sphinxconfiggen
@mock.patch('os.path.isdir')
@mock.patch('os.path.isfile')
@mock.patch('oslo_config.generator.main')
def test_multi_sample_gen(self, main, isfile, isdir):
    isfile.side_effect = [False, True, False, True]
    isdir.return_value = True
    multiple_configs = [('glance-api-gen.conf', 'glance-api'), ('glance-reg-gen.conf', 'glance-reg')]
    config = mock.Mock(config_generator_config_file=multiple_configs)
    app = mock.Mock(srcdir='/opt/glance', config=config)
    sphinxconfiggen.generate_sample(app)
    self.assertEqual(main.call_count, 2)
    main.assert_any_call(args=['--config-file', '/opt/glance/glance-api-gen.conf', '--output-file', '/opt/glance/glance-api.conf.sample'])
    main.assert_any_call(args=['--config-file', '/opt/glance/glance-reg-gen.conf', '--output-file', '/opt/glance/glance-reg.conf.sample'])
from unittest import mock
from oslo_config import fixture
from osprofiler import opts
from osprofiler.tests import test
def test_options_defaults_override(self):
    opts.set_defaults(self.conf_fixture.conf, enabled=True, trace_sqlalchemy=True, hmac_keys='MY_KEY')
    self.assertTrue(self.conf_fixture.conf.profiler.enabled)
    self.assertTrue(self.conf_fixture.conf.profiler.trace_sqlalchemy)
    self.assertEqual('MY_KEY', self.conf_fixture.conf.profiler.hmac_keys)
    self.assertTrue(opts.is_trace_enabled(self.conf_fixture.conf))
    self.assertTrue(opts.is_db_trace_enabled(self.conf_fixture.conf))
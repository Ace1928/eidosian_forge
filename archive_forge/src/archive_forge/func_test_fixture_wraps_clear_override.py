from oslo_config import cfg
from oslo_messaging import conffixture
from oslo_messaging.tests import utils as test_utils
def test_fixture_wraps_clear_override(self):
    conf = self.messaging_conf.conf
    self.assertIsNotNone(conf.clear_override.wrapped)
    self.messaging_conf._teardown_decorator()
    self.assertFalse(hasattr(conf.clear_override, 'wrapped'))
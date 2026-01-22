from glance.hacking import checks
from glance.tests import utils
def test_no_translate_debug_logs(self):
    self.assertEqual(1, len(list(checks.no_translate_debug_logs("LOG.debug(_('foo'))", 'glance/store/foo.py'))))
    self.assertEqual(0, len(list(checks.no_translate_debug_logs("LOG.debug('foo')", 'glance/store/foo.py'))))
    self.assertEqual(0, len(list(checks.no_translate_debug_logs("LOG.info(_('foo'))", 'glance/store/foo.py'))))
from glance.hacking import checks
from glance.tests import utils
def test_no_log_warn(self):
    code = '\n                  LOG.warn("LOG.warn is deprecated")\n               '
    errors = [(1, 0, 'G330')]
    self._assert_has_errors(code, checks.no_log_warn, expected_errors=errors)
    code = '\n                  LOG.warning("LOG.warn is deprecated")\n               '
    self._assert_has_no_errors(code, checks.no_log_warn)
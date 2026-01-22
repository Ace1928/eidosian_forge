from .. import debug, errors, lock, tests
from .scenarios import load_tests_apply_scenarios
def test_create_write_lock(self):
    w_lock = self.write_lock('a-lock-file')
    w_lock.unlock()
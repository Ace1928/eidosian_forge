from breezy import errors, ui
from breezy.tests import per_repository
def test_unlocked(self):
    try:
        self.repo.break_lock()
    except NotImplementedError:
        pass
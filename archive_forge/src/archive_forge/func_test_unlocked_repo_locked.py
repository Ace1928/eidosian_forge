from breezy import errors, ui
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unlocked_repo_locked(self):
    self.workingtree.branch.repository.lock_write()
    ui.ui_factory = ui.CannedInputUIFactory([True])
    try:
        self.unused_workingtree.break_lock()
    except NotImplementedError:
        self.workingtree.branch.repository.unlock()
        return
    if ui.ui_factory.responses == [True]:
        raise TestNotApplicable('repository does not physically lock.')
    self.assertRaises(errors.LockBroken, self.workingtree.branch.repository.unlock)
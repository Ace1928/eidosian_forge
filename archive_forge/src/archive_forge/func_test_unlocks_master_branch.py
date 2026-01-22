from breezy import branch as _mod_branch
from breezy import errors, tests, ui
from breezy.tests import per_branch
def test_unlocks_master_branch(self):
    master = self.make_branch('master')
    try:
        self.branch.bind(master)
    except _mod_branch.BindingUnsupported:
        return
    master.lock_write()
    ui.ui_factory = ui.CannedInputUIFactory([True, True])
    try:
        fresh = _mod_branch.Branch.open(self.unused_branch.base)
        fresh.break_lock()
    except NotImplementedError:
        master.unlock()
        return
    self.assertRaises(errors.LockBroken, master.unlock)
    master.lock_write()
    master.unlock()
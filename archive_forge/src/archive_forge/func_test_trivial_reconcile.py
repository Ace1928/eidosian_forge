from breezy import controldir, tests
from breezy.bzr import inventory
from breezy.repository import WriteGroup
def test_trivial_reconcile(self):
    t = controldir.ControlDir.create_standalone_workingtree('.')
    out, err = self.run_bzr('reconcile')
    if t.branch.repository._reconcile_backsup_inventory:
        does_backup_text = 'Inventory ok.\n'
    else:
        does_backup_text = ''
    self.assertEqualDiff(out, 'Reconciling branch %s\nrevision_history ok.\nReconciling repository %s\n%sReconciliation complete.\n' % (t.branch.base, t.controldir.root_transport.base, does_backup_text))
    self.assertEqualDiff(err, '')
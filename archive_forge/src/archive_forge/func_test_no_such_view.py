from breezy import views as _mod_views
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_no_such_view(self):
    wt = self.make_branch_and_tree('wt')
    try:
        wt.views.lookup_view('opaque')
    except _mod_views.NoSuchView as e:
        self.assertEqual(e.view_name, 'opaque')
        self.assertEqual(str(e), 'No such view: opaque.')
    else:
        self.fail("didn't get expected exception")
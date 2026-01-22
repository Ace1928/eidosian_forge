from breezy import views as _mod_views
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_set_view(self):
    wt = self.make_branch_and_tree('wt')
    wt.views.set_view('view-1', ['dir-1'])
    current, views = wt.views.get_view_info()
    self.assertEqual('view-1', current)
    self.assertEqual({'view-1': ['dir-1']}, views)
    wt.views.set_view('view-2', ['dir-2'], make_current=False)
    current, views = wt.views.get_view_info()
    self.assertEqual('view-1', current)
    self.assertEqual({'view-1': ['dir-1'], 'view-2': ['dir-2']}, views)
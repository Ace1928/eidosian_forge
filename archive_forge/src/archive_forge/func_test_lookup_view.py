from breezy import views as _mod_views
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
def test_lookup_view(self):
    wt = self.make_branch_and_tree('wt')
    view_current = 'view-name'
    view_dict = {view_current: ['dir-1'], 'other-name': ['dir-2']}
    wt.views.set_view_info(view_current, view_dict)
    result = wt.views.lookup_view()
    self.assertEqual(result, ['dir-1'])
    result = wt.views.lookup_view('other-name')
    self.assertEqual(result, ['dir-2'])
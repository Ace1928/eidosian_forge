from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_docs(self):
    doc = 'Invoked after changing the tip of a branch object. Called with a breezy.branch.PostChangeBranchTipParams object'
    hook = HookPoint('post_tip_change', doc, (0, 15), None)
    self.assertEqual('post_tip_change\n~~~~~~~~~~~~~~~\n\nIntroduced in: 0.15\n\nInvoked after changing the tip of a branch object. Called with a\nbreezy.branch.PostChangeBranchTipParams object\n', hook.docs())
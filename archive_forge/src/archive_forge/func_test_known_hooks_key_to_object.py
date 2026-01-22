from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_known_hooks_key_to_object(self):
    self.assertIs(branch.Branch.hooks, known_hooks_key_to_object(('breezy.branch', 'Branch.hooks')))
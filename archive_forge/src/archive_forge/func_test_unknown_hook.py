from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_unknown_hook(self):
    error = UnknownHook('branch', 'foo')
    self.assertEqualDiff("The branch hook 'foo' is unknown in this version of breezy.", str(error))
    error = UnknownHook('tree', 'bar')
    self.assertEqualDiff("The tree hook 'bar' is unknown in this version of breezy.", str(error))
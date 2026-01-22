from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_install_named_hook_lazy_old(self):
    hooks = Hooks('breezy.tests.hooks', 'some_hooks')
    hooks['set_rh'] = []
    self.assertRaises(errors.UnsupportedOperation, hooks.install_named_hook_lazy, 'set_rh', 'breezy.tests.test_hooks', 'TestHooks.set_rh', 'demo')
from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_uninstall_unknown(self):
    hook = HookPoint('foo', 'no docs', None, None)
    self.assertRaises(KeyError, hook.uninstall, 'my callback')
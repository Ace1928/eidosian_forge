from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_valid_lazy_hooks(self):
    for key, callbacks in _mod_hooks._lazy_hooks.items():
        module_name, member_name, hook_name = key
        obj = pyutils.get_named_object(module_name, member_name)
        self.assertEqual(obj._module, module_name)
        self.assertEqual(obj._member_name, member_name)
        self.assertTrue(hook_name in obj)
        self.assertIs(callbacks, obj[hook_name]._callbacks)
import os
import tempfile
import testtools
from troveclient import utils
def test_run_hookable_mixin(self):
    hook_type = 'hook_type'
    mixin = utils.HookableMixin()
    mixin.add_hook(hook_type, self.func)
    mixin.run_hooks(hook_type)
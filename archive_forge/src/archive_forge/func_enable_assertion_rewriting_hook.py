from __future__ import (absolute_import, division, print_function)
import os
def enable_assertion_rewriting_hook():
    """
    Enable pytest's AssertionRewritingHook on Python 3.x.
    This is necessary because the Ansible collection loader intercepts imports before the pytest provided loader ever sees them.
    """
    import sys
    if sys.version_info[0] == 2:
        return
    hook_name = '_pytest.assertion.rewrite.AssertionRewritingHook'
    hooks = [hook for hook in sys.meta_path if hook.__class__.__module__ + '.' + hook.__class__.__qualname__ == hook_name]
    if len(hooks) != 1:
        raise Exception('Found {} instance(s) of "{}" in sys.meta_path.'.format(len(hooks), hook_name))
    assertion_rewriting_hook = hooks[0]

    def exec_module(self, module):
        if self._redirect_module:
            return
        code_obj = self.get_code(self._fullname)
        if code_obj is not None:
            should_rewrite = self._package_to_load == 'conftest' or self._package_to_load.startswith('test_')
            if should_rewrite:
                assertion_rewriting_hook.exec_module(module)
            else:
                exec(code_obj, module.__dict__)
    from ansible.utils.collection_loader._collection_finder import _AnsibleCollectionPkgLoaderBase
    _AnsibleCollectionPkgLoaderBase.exec_module = exec_module
from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
def install_fusion_exception_hook(module):
    """Installs a hook that catches `purefusion.rest.ApiException` and
    `OperationException` and produces simpler and nicer error messages
    for Ansible output."""
    original_hook = sys.excepthook
    sys.excepthook = lambda type, value, traceback: _except_hook_callback(module, original_hook, type, value, traceback)
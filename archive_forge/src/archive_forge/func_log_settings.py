import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from torch._dynamo.utils import LazyString
import logging
def log_settings(settings):
    exit_stack = contextlib.ExitStack()
    settings_patch = unittest.mock.patch.dict(os.environ, {'TORCH_LOGS': settings})
    exit_stack.enter_context(preserve_log_state())
    exit_stack.enter_context(settings_patch)
    torch._logging._internal._init_logs()
    return exit_stack
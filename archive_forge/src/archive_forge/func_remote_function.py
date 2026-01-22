import contextlib
import os
from modin.error_message import ErrorMessage
def remote_function(func, ignore_defaults=False):
    return func
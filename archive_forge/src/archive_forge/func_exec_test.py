import contextlib
import os
import platform
import socket
import sys
import textwrap
import typing  # noqa: F401
import unittest
import warnings
from tornado.testing import bind_unused_port
def exec_test(caller_globals, caller_locals, s):
    """Execute ``s`` in a given context and return the result namespace.

    Used to define functions for tests in particular python
    versions that would be syntax errors in older versions.
    """
    global_namespace = dict(caller_globals, **caller_locals)
    local_namespace = {}
    exec(textwrap.dedent(s), global_namespace, local_namespace)
    return local_namespace
import functools
import gc
import os
import platform
import re
import socket
import subprocess
import sys
import time
def assert_python_ok(*args, **env_vars):
    """
    Assert that running the interpreter with `args` and optional environment
    variables `env_vars` succeeds (rc == 0) and return a (return code, stdout,
    stderr) tuple.

    If the __cleanenv keyword is set, env_vars is used a fresh environment.

    Python is started in isolated mode (command line option -I),
    except if the __isolated keyword is set to False.
    """
    return _assert_python(True, *args, **env_vars)
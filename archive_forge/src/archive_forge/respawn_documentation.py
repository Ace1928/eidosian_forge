from __future__ import (absolute_import, division, print_function)
import os
import subprocess
import sys
from ansible.module_utils.common.text.converters import to_bytes
import runpy
import sys

    Probes a supplied list of Python interpreters, returning the first one capable of
    importing the named module. This is useful when attempting to locate a "system
    Python" where OS-packaged utility modules are located.

    :arg interpreter_paths: iterable of paths to Python interpreters. The paths will be probed
    in order, and the first path that exists and can successfully import the named module will
    be returned (or ``None`` if probing fails for all supplied paths).
    :arg module_name: fully-qualified Python module name to probe for (eg, ``selinux``)
    
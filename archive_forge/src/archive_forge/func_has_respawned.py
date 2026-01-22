from __future__ import (absolute_import, division, print_function)
import os
import subprocess
import sys
from ansible.module_utils.common.text.converters import to_bytes
import runpy
import sys
def has_respawned():
    return hasattr(sys.modules['__main__'], '_respawned')
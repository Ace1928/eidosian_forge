from __future__ import (absolute_import, division, print_function)
import resource
import base64
import contextlib
import errno
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
def common_pip_environment():
    """Return common environment variables used to run pip."""
    env = os.environ.copy()
    return env
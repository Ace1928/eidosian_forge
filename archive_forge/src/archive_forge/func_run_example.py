import os
import subprocess
import sys
import tempfile
import example_utils  # noqa
def run_example(name, add_env=None):
    _exec([sys.executable, _path_to(name), backend_uri], add_env)
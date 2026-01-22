import os
import tempfile
import textwrap
import unittest
from bpython import config
def load_temp_config(self, content):
    """Write config to a temporary file and load it."""
    with tempfile.NamedTemporaryFile() as f:
        f.write(content.encode('utf8'))
        f.flush()
        return config.Config(f.name)
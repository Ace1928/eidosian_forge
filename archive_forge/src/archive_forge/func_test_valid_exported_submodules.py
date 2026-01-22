import os
import shutil
import sys
import tempfile
import unittest
from os.path import join
from tempfile import TemporaryDirectory
from IPython.core.completerlib import magic_run_completer, module_completion, try_import
from IPython.testing.decorators import onlyif_unicode_paths
def test_valid_exported_submodules():
    """
    Test checking exported (__all__) objects are submodules
    """
    results = module_completion('import os.pa')
    assert 'os.path' in results
    assert 'os.pathconf' not in results
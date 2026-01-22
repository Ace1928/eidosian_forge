import shutil
import sys
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
import pytest
from IPython.core.profileapp import list_bundled_profiles, list_profiles_in
from IPython.core.profiledir import ProfileDir
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.process import getoutput
def test_profile_create_ipython_dir():
    """ipython profile create respects --ipython-dir"""
    with TemporaryDirectory() as td:
        getoutput([sys.executable, '-m', 'IPython', 'profile', 'create', 'foo', '--ipython-dir=%s' % td])
        profile_dir = Path(td) / 'profile_foo'
        assert Path(profile_dir).exists()
        ipython_config = profile_dir / 'ipython_config.py'
        assert Path(ipython_config).exists()
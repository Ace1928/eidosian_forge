import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def get_subpackage(self, subpackage_name, subpackage_path=None, parent_name=None, caller_level=1):
    """Return list of subpackage configurations.

        Parameters
        ----------
        subpackage_name : str or None
            Name of the subpackage to get the configuration. '*' in
            subpackage_name is handled as a wildcard.
        subpackage_path : str
            If None, then the path is assumed to be the local path plus the
            subpackage_name. If a setup.py file is not found in the
            subpackage_path, then a default configuration is used.
        parent_name : str
            Parent name.
        """
    if subpackage_name is None:
        if subpackage_path is None:
            raise ValueError('either subpackage_name or subpackage_path must be specified')
        subpackage_name = os.path.basename(subpackage_path)
    l = subpackage_name.split('.')
    if subpackage_path is None and '*' in subpackage_name:
        return self._wildcard_get_subpackage(subpackage_name, parent_name, caller_level=caller_level + 1)
    assert '*' not in subpackage_name, repr((subpackage_name, subpackage_path, parent_name))
    if subpackage_path is None:
        subpackage_path = njoin([self.local_path] + l)
    else:
        subpackage_path = njoin([subpackage_path] + l[:-1])
        subpackage_path = self.paths([subpackage_path])[0]
    setup_py = njoin(subpackage_path, self.setup_name)
    if not self.options['ignore_setup_xxx_py']:
        if not os.path.isfile(setup_py):
            setup_py = njoin(subpackage_path, 'setup_%s.py' % subpackage_name)
    if not os.path.isfile(setup_py):
        if not self.options['assume_default_configuration']:
            self.warn('Assuming default configuration (%s/{setup_%s,setup}.py was not found)' % (os.path.dirname(setup_py), subpackage_name))
        config = Configuration(subpackage_name, parent_name, self.top_path, subpackage_path, caller_level=caller_level + 1)
    else:
        config = self._get_configuration_from_setup_py(setup_py, subpackage_name, subpackage_path, parent_name, caller_level=caller_level + 1)
    if config:
        return [config]
    else:
        return []
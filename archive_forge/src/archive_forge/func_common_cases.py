import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib import reload
from os.path import abspath, join
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
import IPython
from IPython import paths
from IPython.testing import decorators as dec
from IPython.testing.decorators import (
from IPython.testing.tools import make_tempfile
from IPython.utils import path
def common_cases(self):
    return [(['*'], self.filenames), (['a*'], self.filenames_start_with_a), (['*c'], ['*c']), (['*', 'a*', '*b', '*c'], self.filenames + self.filenames_start_with_a + self.filenames_end_with_b + ['*c']), (['a[012]'], self.filenames_start_with_a)]
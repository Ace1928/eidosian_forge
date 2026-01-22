import sys
import os
import importlib.util
import unittest
from distutils.command.install_lib import install_lib
from distutils.extension import Extension
from distutils.tests import support
from distutils.errors import DistutilsOptionError
from test.support import requires_subprocess
Tests for distutils.command.install_data.
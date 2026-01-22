import os
import sys
import zipfile
import unittest
from distutils.core import Distribution
from distutils.command.bdist_dumb import bdist_dumb
from distutils.tests import support
from distutils.core import setup
import foo
Tests for distutils.command.bdist_dumb.
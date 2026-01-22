from distutils.util import convert_path
from distutils import log
from distutils.errors import DistutilsOptionError
import distutils
import os
import configparser
from setuptools import Command
Save command-line options to a file
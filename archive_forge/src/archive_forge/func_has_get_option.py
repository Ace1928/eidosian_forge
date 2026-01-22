import logging  # noqa
from collections import defaultdict
import io
import os
import re
import shlex
import sys
import traceback
import distutils.ccompiler
from distutils import errors
from distutils import log
import pkg_resources
from setuptools import dist as st_dist
from setuptools import extension
from pbr import extra_files
import pbr.hooks
def has_get_option(config, section, option):
    if section in config and option in config[section]:
        return config[section][option]
    else:
        return False
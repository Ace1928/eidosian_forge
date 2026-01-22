from glob import glob
from distutils.util import get_platform
from distutils.util import convert_path, subst_vars
from distutils.errors import (
from distutils import log, dir_util
from distutils.command.build_scripts import first_line_re
from distutils.spawn import find_executable
from distutils.command import install
import sys
import os
from typing import Dict, List
import zipimport
import shutil
import tempfile
import zipfile
import re
import stat
import random
import textwrap
import warnings
import site
import struct
import contextlib
import subprocess
import shlex
import io
import configparser
import sysconfig
from sysconfig import get_path
from setuptools import Command
from setuptools.sandbox import run_setup
from setuptools.command import setopt
from setuptools.archive_util import unpack_archive
from setuptools.package_index import (
from setuptools.command import bdist_egg, egg_info
from setuptools.warnings import SetuptoolsDeprecationWarning, SetuptoolsWarning
from setuptools.wheel import Wheel
from pkg_resources import (
import pkg_resources
from ..compat import py39, py311
from .._path import ensure_directory
from ..extern.jaraco.text import yield_lines
def process_distribution(self, requirement, dist, deps=True, *info):
    self.update_pth(dist)
    self.package_index.add(dist)
    if dist in self.local_index[dist.key]:
        self.local_index.remove(dist)
    self.local_index.add(dist)
    self.install_egg_scripts(dist)
    self.installed_projects[dist.key] = dist
    log.info(self.installation_report(requirement, dist, *info))
    if dist.has_metadata('dependency_links.txt') and (not self.no_find_links):
        self.package_index.add_find_links(dist.get_metadata_lines('dependency_links.txt'))
    if not deps and (not self.always_copy):
        return
    elif requirement is not None and dist.key != requirement.key:
        log.warn('Skipping dependencies for %s', dist)
        return
    elif requirement is None or dist not in requirement:
        distreq = dist.as_requirement()
        requirement = Requirement(str(distreq))
    log.info('Processing dependencies for %s', requirement)
    try:
        distros = WorkingSet([]).resolve([requirement], self.local_index, self.easy_install)
    except DistributionNotFound as e:
        raise DistutilsError(str(e)) from e
    except VersionConflict as e:
        raise DistutilsError(e.report()) from e
    if self.always_copy or self.always_copy_from:
        for dist in distros:
            if dist.key not in self.installed_projects:
                self.easy_install(dist.as_requirement())
    log.info('Finished processing dependencies for %s', requirement)
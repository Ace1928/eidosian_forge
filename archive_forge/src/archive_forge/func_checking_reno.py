from __future__ import unicode_literals
from distutils.command import install as du_install
from distutils import log
import email
import email.errors
import os
import re
import sys
import warnings
import pkg_resources
import setuptools
from setuptools.command import develop
from setuptools.command import easy_install
from setuptools.command import egg_info
from setuptools.command import install
from setuptools.command import install_scripts
from setuptools.command import sdist
from pbr import extra_files
from pbr import git
from pbr import options
import pbr.pbr_json
from pbr import testr_command
from pbr import version
import threading
from %(module_name)s import %(import_target)s
import sys
from %(module_name)s import %(import_target)s
def checking_reno(self):
    """Ensure reno is installed and configured.

        We can't run reno-based commands if reno isn't installed/available, and
        don't want to if the user isn't using it.
        """
    if hasattr(self, '_has_reno'):
        return self._has_reno
    option_dict = self.distribution.get_option_dict('pbr')
    should_skip = options.get_boolean_option(option_dict, 'skip_reno', 'SKIP_GENERATE_RENO')
    if should_skip:
        self._has_reno = False
        return False
    try:
        from reno import setup_command
    except ImportError:
        log.info('[pbr] reno was not found or is too old. Skipping release notes')
        self._has_reno = False
        return False
    conf, output_file, cache_file = setup_command.load_config(self.distribution)
    if not os.path.exists(os.path.join(conf.reporoot, conf.notespath)):
        log.info('[pbr] reno does not appear to be configured. Skipping release notes')
        self._has_reno = False
        return False
    self._files = [output_file, cache_file]
    log.info('[pbr] Generating release notes')
    self._has_reno = True
    return True
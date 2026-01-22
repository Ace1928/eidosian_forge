from distutils.util import convert_path
from distutils import log
from distutils.errors import DistutilsOptionError
import os
import glob
import io
from setuptools.command.easy_install import easy_install
from setuptools import _path
from setuptools import namespaces
import setuptools
def uninstall_link(self):
    if os.path.exists(self.egg_link):
        log.info('Removing %s (link to %s)', self.egg_link, self.egg_base)
        egg_link_file = open(self.egg_link)
        contents = [line.rstrip() for line in egg_link_file]
        egg_link_file.close()
        if contents not in ([self.egg_path], [self.egg_path, self.setup_path]):
            log.warn('Link points to %s: uninstall aborted', contents)
            return
        if not self.dry_run:
            os.unlink(self.egg_link)
    if not self.dry_run:
        self.update_pth(self.dist)
    if self.distribution.scripts:
        log.warn('Note: you must uninstall or replace scripts manually!')
import fnmatch
import os
import platform
import re
import sys
from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution
def mkdir_and_copy_file(self, header):
    install_dir = os.path.join(self.install_dir, os.path.dirname(header))
    install_dir = re.sub('/google/protobuf_archive/src', '', install_dir)
    external_header_locations = ['tensorflow/include/external/eigen_archive/', 'tensorflow/include/external/com_google_absl/']
    for location in external_header_locations:
        if location in install_dir:
            extra_dir = install_dir.replace(location, '')
            if not os.path.exists(extra_dir):
                self.mkpath(extra_dir)
            self.copy_file(header, extra_dir)
    if not os.path.exists(install_dir):
        self.mkpath(install_dir)
    return self.copy_file(header, install_dir)
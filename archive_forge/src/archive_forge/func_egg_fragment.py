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
def egg_fragment(match):
    return re.sub('(?P<PackageName>[\\w.-]+)-(?P<GlobalVersion>(?P<VersionTripple>(?P<Major>0|[1-9][0-9]*)\\.(?P<Minor>0|[1-9][0-9]*)\\.(?P<Patch>0|[1-9][0-9]*)){1}(?P<Tags>(?:\\-(?P<Prerelease>(?:(?=[0]{1}[0-9A-Za-z-]{0})(?:[0]{1})|(?=[1-9]{1}[0-9]*[A-Za-z]{0})(?:[0-9]+)|(?=[0-9]*[A-Za-z-]+[0-9A-Za-z-]*)(?:[0-9A-Za-z-]+)){1}(?:\\.(?=[0]{1}[0-9A-Za-z-]{0})(?:[0]{1})|\\.(?=[1-9]{1}[0-9]*[A-Za-z]{0})(?:[0-9]+)|\\.(?=[0-9]*[A-Za-z-]+[0-9A-Za-z-]*)(?:[0-9A-Za-z-]+))*){1}){0,1}(?:\\+(?P<Meta>(?:[0-9A-Za-z-]+(?:\\.[0-9A-Za-z-]+)*))){0,1}))', '\\g<PackageName>>=\\g<GlobalVersion>', match.groups()[-1])
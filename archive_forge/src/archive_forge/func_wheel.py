from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import os
import datetime
import traceback
import platform  # NOQA
from _ast import *  # NOQA
from ast import parse  # NOQA
from setuptools import setup, Extension, Distribution  # NOQA
from setuptools.command import install_lib  # NOQA
from setuptools.command.sdist import sdist as _sdist  # NOQA
def wheel(self, kw, setup):
    """temporary add setup.cfg if creating a wheel to include LICENSE file
        https://bitbucket.org/pypa/wheel/issues/47
        """
    if 'bdist_wheel' not in sys.argv:
        return False
    file_name = 'setup.cfg'
    if os.path.exists(file_name):
        return False
    with open(file_name, 'w') as fp:
        if os.path.exists('LICENSE'):
            fp.write('[metadata]\nlicense-file = LICENSE\n')
        else:
            print('\n\n>>>>>> LICENSE file not found <<<<<\n\n')
        if self._pkg_data.get('universal'):
            fp.write('[bdist_wheel]\nuniversal = 1\n')
    try:
        setup(**kw)
    except Exception:
        raise
    finally:
        os.remove(file_name)
    return True
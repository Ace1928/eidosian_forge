import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
def get_option_single(self, *options):
    """ Ensure that only one of `options` are found in the section

        Parameters
        ----------
        *options : list of str
           a list of options to be found in the section (``self.section``)

        Returns
        -------
        str :
            the option that is uniquely found in the section

        Raises
        ------
        AliasedOptionError :
            in case more than one of the options are found
        """
    found = [self.cp.has_option(self.section, opt) for opt in options]
    if sum(found) == 1:
        return options[found.index(True)]
    elif sum(found) == 0:
        return options[0]
    if AliasedOptionError.__doc__ is None:
        raise AliasedOptionError()
    raise AliasedOptionError(AliasedOptionError.__doc__.format(section=self.section, options='[{}]'.format(', '.join(options))))
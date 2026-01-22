import argparse
import collections
from collections import abc
import copy
import enum
import errno
import functools
import glob
import inspect
import itertools
import logging
import os
import string
import sys
from oslo_config import iniparser
from oslo_config import sources
import oslo_config.sources._environment as _environment
from oslo_config import types
import stevedore
def find_config_dirs(project=None, prog=None, extension='.conf.d'):
    """Return a list of default configuration dirs.

    :param project: an optional project name
    :param prog: the program name, defaulting to the basename of
        sys.argv[0], without extension .py
    :param extension: the type of the config directory. Defaults to '.conf.d'

    We default to two config dirs: [${project}.conf.d/, ${prog}.conf.d/].
    If no project name is supplied, we only look for ${prog.conf.d/}.

    And we look for those config dirs in the following directories::

      ~/.${project}/
      ~/
      /etc/${project}/
      /etc/
      ${SNAP_COMMON}/etc/${project}
      ${SNAP}/etc/${project}

    We return an absolute path for each of the two config dirs,
    in the first place we find it (iff we find it).

    For example, if project=foo, prog=bar and /etc/foo/foo.conf.d/,
    /etc/bar.conf.d/ and ~/.foo/bar.conf.d/ all exist, then we return
    ['/etc/foo/foo.conf.d/', '~/.foo/bar.conf.d/']
    """
    return _find_config_files(project, prog, extension)
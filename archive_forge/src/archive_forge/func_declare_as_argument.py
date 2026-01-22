import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
def declare_as_argument(self, *args, **kwds):
    """Map this Config item to an argparse argument.

        Valid arguments include all valid arguments to argparse's
        ArgumentParser.add_argument() with the exception of 'default'.
        In addition, you may provide a group keyword argument to either
        pass in a pre-defined option group or subparser, or else pass in
        the string name of a group, subparser, or (subparser, group).

        """
    if 'default' in kwds:
        raise TypeError('You cannot specify an argparse default value with ConfigBase.declare_as_argument().  The default value is supplied automatically from the Config definition.')
    if 'action' not in kwds and self._domain is bool:
        if not self._default:
            kwds['action'] = 'store_true'
        else:
            kwds['action'] = 'store_false'
            if not args:
                args = ('--disable-' + _munge_name(self.name()),)
            if 'help' not in kwds:
                kwds['help'] = "[DON'T] " + self._description
    if 'help' not in kwds:
        kwds['help'] = self._description
    if not args:
        args = ('--' + _munge_name(self.name()),)
    if self._argparse:
        self._argparse = self._argparse + ((args, kwds),)
    else:
        self._argparse = ((args, kwds),)
    return self
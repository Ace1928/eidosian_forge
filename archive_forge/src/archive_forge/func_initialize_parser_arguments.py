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
def initialize_parser_arguments(self):
    for container, values in self._args_cache.items():
        index = 0
        has_positional = False
        for index, argument in enumerate(values):
            if not argument['args'][0].startswith('-'):
                has_positional = True
                break
        size = index if has_positional else len(values)
        values[:size] = sorted(values[:size], key=lambda x: x['args'])
        for argument in values:
            try:
                container.add_argument(*argument['args'], **argument['kwargs'])
            except argparse.ArgumentError:
                options = ','.join(argument['args'])
                raise DuplicateOptError(options)
    self._args_cache = {}
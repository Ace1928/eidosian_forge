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
@__clear_cache
def set_override(self, name, override, group=None):
    """Override an opt value.

        Override the command line, config file and default values of a
        given option.

        :param name: the name/dest of the opt
        :param override: the override value
        :param group: an option OptGroup object or group name

        :raises: NoSuchOptError, NoSuchGroupError
        """
    opt_info = self._get_opt_info(name, group)
    opt_info['override'] = self._get_enforced_type_value(opt_info['opt'], override)
    opt_info['location'] = LocationInfo(Locations.set_override, _get_caller_detail(3))
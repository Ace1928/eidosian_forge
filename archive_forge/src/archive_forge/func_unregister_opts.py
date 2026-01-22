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
def unregister_opts(self, opts, group=None):
    """Unregister multiple CLI option schemas at once."""
    for opt in opts:
        self.unregister_opt(opt, group, clear_cache=False)
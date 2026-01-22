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
def mutate_config_files(self):
    """Reload configure files and parse all options.

        Only options marked as 'mutable' will appear to change.

        Hooks are called in a NON-DETERMINISTIC ORDER. Do not expect hooks to
        be called in the same order as they were added.

        :return: {(None or 'group', 'optname'): (old_value, new_value), ... }
        :raises: Error if reloading fails
        """
    self.__cache.clear()
    old_mutate_ns = self._mutable_ns or self._namespace
    self._mutable_ns = self._reload_config_files()
    self._warn_immutability()
    fresh = self._diff_ns(old_mutate_ns, self._mutable_ns)

    def key_fn(item):
        groupname, optname = item[0]
        return item[0] if groupname else ('\t', optname)
    sorted_fresh = sorted(fresh.items(), key=key_fn)
    for (groupname, optname), (old, new) in sorted_fresh:
        groupname = groupname if groupname else 'DEFAULT'
        LOG.info('Option %(group)s.%(option)s changed from [%(old_val)s] to [%(new_val)s]', {'group': groupname, 'option': optname, 'old_val': old, 'new_val': new})
    for hook in self._mutate_hooks:
        hook(self, fresh)
    return fresh
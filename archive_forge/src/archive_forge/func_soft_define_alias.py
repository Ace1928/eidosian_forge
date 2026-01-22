import os
import re
import sys
from traitlets.config.configurable import Configurable
from .error import UsageError
from traitlets import List, Instance
from logging import error
import typing as t
def soft_define_alias(self, name, cmd):
    """Define an alias, but don't raise on an AliasError."""
    try:
        self.define_alias(name, cmd)
    except AliasError as e:
        error('Invalid alias: %s' % e)
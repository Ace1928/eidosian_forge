from __future__ import annotations
import functools
import json
import logging
import os
import pprint
import re
import sys
import typing as t
from collections import OrderedDict, defaultdict
from contextlib import suppress
from copy import deepcopy
from logging.config import dictConfig
from textwrap import dedent
from traitlets.config.configurable import Configurable, SingletonConfigurable
from traitlets.config.loader import (
from traitlets.traitlets import (
from traitlets.utils.bunch import Bunch
from traitlets.utils.nested_update import nested_update
from traitlets.utils.text import indent, wrap_paragraphs
from ..utils import cast_unicode
from ..utils.importstring import import_item
def emit_help(self, classes: bool=False) -> t.Generator[str, None, None]:
    """Yield the help-lines for each Configurable class in self.classes.

        If classes=False (the default), only flags and aliases are printed.
        """
    yield from self.emit_description()
    yield from self.emit_subcommands_help()
    yield from self.emit_options_help()
    if classes:
        help_classes = self._classes_with_config_traits()
        if help_classes is not None:
            yield 'Class options'
            yield '============='
            for p in wrap_paragraphs(self.keyvalue_description):
                yield p
                yield ''
        for cls in help_classes:
            yield cls.class_get_help()
            yield ''
    yield from self.emit_examples()
    yield from self.emit_help_epilogue(classes)
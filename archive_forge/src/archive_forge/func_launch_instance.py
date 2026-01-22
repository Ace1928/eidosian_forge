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
@classmethod
def launch_instance(cls, argv: ArgvType=None, **kwargs: t.Any) -> None:
    """Launch a global instance of this Application

        If a global instance already exists, this reinitializes and starts it
        """
    app = cls.instance(**kwargs)
    app.initialize(argv)
    app.start()
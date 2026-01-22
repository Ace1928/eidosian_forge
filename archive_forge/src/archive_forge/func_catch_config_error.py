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
def catch_config_error(method: T) -> T:
    """Method decorator for catching invalid config (Trait/ArgumentErrors) during init.

    On a TraitError (generally caused by bad config), this will print the trait's
    message, and exit the app.

    For use on init methods, to prevent invoking excepthook on invalid input.
    """

    @functools.wraps(method)
    def inner(app: Application, *args: t.Any, **kwargs: t.Any) -> t.Any:
        try:
            return method(app, *args, **kwargs)
        except (TraitError, ArgumentError) as e:
            app.log.fatal('Bad config encountered during initialization: %s', e)
            app.log.debug('Config at the time: %s', app.config)
            app.exit(1)
    return t.cast(T, inner)
import re
import traceback
import types
from collections import OrderedDict
from os import getenv, path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Generator, Iterator, List, NamedTuple,
from sphinx.errors import ConfigError, ExtensionError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.i18n import format_date
from sphinx.util.osutil import cd, fs_encoding
from sphinx.util.tags import Tags
from sphinx.util.typing import NoneType
def check_confval_types(app: Optional['Sphinx'], config: Config) -> None:
    """Check all values for deviation from the default value's type, since
    that can result in TypeErrors all over the place NB.
    """
    for confval in config:
        default, rebuild, annotations = config.values[confval.name]
        if callable(default):
            default = default(config)
        if default is None and (not annotations):
            continue
        if annotations is Any:
            pass
        elif isinstance(annotations, ENUM):
            if not annotations.match(confval.value):
                msg = __('The config value `{name}` has to be a one of {candidates}, but `{current}` is given.')
                logger.warning(msg.format(name=confval.name, current=confval.value, candidates=annotations.candidates), once=True)
        else:
            if type(confval.value) is type(default):
                continue
            if type(confval.value) in annotations:
                continue
            common_bases = set(type(confval.value).__bases__ + (type(confval.value),)) & set(type(default).__bases__)
            common_bases.discard(object)
            if common_bases:
                continue
            if annotations:
                msg = __("The config value `{name}' has type `{current.__name__}'; expected {permitted}.")
                wrapped_annotations = ["`{}'".format(c.__name__) for c in annotations]
                if len(wrapped_annotations) > 2:
                    permitted = '{}, or {}'.format(', '.join(wrapped_annotations[:-1]), wrapped_annotations[-1])
                else:
                    permitted = ' or '.join(wrapped_annotations)
                logger.warning(msg.format(name=confval.name, current=type(confval.value), permitted=permitted), once=True)
            else:
                msg = __("The config value `{name}' has type `{current.__name__}', defaults to `{default.__name__}'.")
                logger.warning(msg.format(name=confval.name, current=type(confval.value), default=type(default)), once=True)
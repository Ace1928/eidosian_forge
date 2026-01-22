import collections
import inspect
import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union
from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.inspect import stringify_annotation
from sphinx.util.typing import get_type_hints
def convert_obj(obj, translations, default_translation):
    translation = translations.get(obj, obj)
    if translation in _SINGLETONS and default_translation == ':class:`%s`':
        default_translation = ':obj:`%s`'
    elif translation == '...' and default_translation == ':class:`%s`':
        default_translation = ':obj:`%s <Ellipsis>`'
    if _xref_regex.match(translation) is None:
        translation = default_translation % translation
    return translation
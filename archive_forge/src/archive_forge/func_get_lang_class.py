import functools
import importlib
import importlib.util
import inspect
import itertools
import logging
import os
import pkgutil
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import (
import catalogue
import langcodes
import numpy
import srsly
import thinc
from catalogue import Registry, RegistryError
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from thinc.api import (
from thinc.api import compounding, decaying, fix_random_seed  # noqa: F401
from . import about
from .compat import CudaStream, cupy, importlib_metadata, is_windows
from .errors import OLD_MODEL_SHORTCUTS, Errors, Warnings
from .symbols import ORTH
def get_lang_class(lang: str) -> Type['Language']:
    """Import and load a Language class.

    lang (str): IETF language code, such as 'en'.
    RETURNS (Language): Language class.
    """
    if lang in registry.languages:
        return registry.languages.get(lang)
    else:
        try:
            module = importlib.import_module(f'.lang.{lang}', 'spacy')
        except ImportError as err:
            try:
                match = find_matching_language(lang)
            except langcodes.tag_parser.LanguageTagError:
                match = None
            if match:
                lang = match
                module = importlib.import_module(f'.lang.{lang}', 'spacy')
            else:
                raise ImportError(Errors.E048.format(lang=lang, err=err)) from err
        set_lang_class(lang, getattr(module, module.__all__[0]))
    return registry.languages.get(lang)
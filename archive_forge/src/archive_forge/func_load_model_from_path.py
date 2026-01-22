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
def load_model_from_path(model_path: Path, *, meta: Optional[Dict[str, Any]]=None, vocab: Union['Vocab', bool]=True, disable: Union[str, Iterable[str]]=_DEFAULT_EMPTY_PIPES, enable: Union[str, Iterable[str]]=_DEFAULT_EMPTY_PIPES, exclude: Union[str, Iterable[str]]=_DEFAULT_EMPTY_PIPES, config: Union[Dict[str, Any], Config]=SimpleFrozenDict()) -> 'Language':
    """Load a model from a data directory path. Creates Language class with
    pipeline from config.cfg and then calls from_disk() with path.

    model_path (Path): Model path.
    meta (Dict[str, Any]): Optional model meta.
    vocab (Vocab / True): Optional vocab to pass in on initialization. If True,
        a new Vocab object will be created.
    disable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to disable. Disabled
        pipes will be loaded but they won't be run unless you explicitly
        enable them by calling nlp.enable_pipe.
    enable (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to enable. All other
        pipes will be disabled (and can be enabled using `nlp.enable_pipe`).
    exclude (Union[str, Iterable[str]]): Name(s) of pipeline component(s) to exclude. Excluded
        components won't be loaded.
    config (Dict[str, Any] / Config): Config overrides as nested dict or dict
        keyed by section values in dot notation.
    RETURNS (Language): The loaded nlp object.
    """
    if not model_path.exists():
        raise IOError(Errors.E052.format(path=model_path))
    if not meta:
        meta = get_model_meta(model_path)
    config_path = model_path / 'config.cfg'
    overrides = dict_to_dot(config, for_overrides=True)
    config = load_config(config_path, overrides=overrides)
    nlp = load_model_from_config(config, vocab=vocab, disable=disable, enable=enable, exclude=exclude, meta=meta)
    return nlp.from_disk(model_path, exclude=exclude, overrides=overrides)
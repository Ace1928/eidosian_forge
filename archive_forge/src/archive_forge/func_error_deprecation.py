from __future__ import annotations
import sys
import os
import warnings
import glob
from importlib import import_module
import ruamel.yaml
from ruamel.yaml.error import UnsafeLoaderWarning, YAMLError  # NOQA
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.nodes import *  # NOQA
from ruamel.yaml.loader import BaseLoader, SafeLoader, Loader, RoundTripLoader  # NOQA
from ruamel.yaml.dumper import BaseDumper, SafeDumper, Dumper, RoundTripDumper  # NOQA
from ruamel.yaml.compat import StringIO, BytesIO, with_metaclass, nprint, nprintf  # NOQA
from ruamel.yaml.resolver import VersionedResolver, Resolver  # NOQA
from ruamel.yaml.representer import (
from ruamel.yaml.constructor import (
from ruamel.yaml.loader import Loader as UnsafeLoader  # NOQA
from ruamel.yaml.comments import CommentedMap, CommentedSeq, C_PRE
from ruamel.yaml.docinfo import DocInfo, version, Version
def error_deprecation(fun: Any, method: Any, arg: str='', comment: str='instead of') -> None:
    import inspect
    s = f'\n"{fun}()" has been removed, use\n\n  yaml = YAML({arg})\n  yaml.{method}(...)\n\n{comment}'
    try:
        info = inspect.getframeinfo(inspect.stack()[2][0])
        context = '' if info.code_context is None else ''.join(info.code_context)
        s += f' file "{info.filename}", line {info.lineno}\n\n{context}'
    except Exception as e:
        _ = e
    s += '\n'
    if sys.version_info < (3, 10):
        raise AttributeError(s)
    else:
        raise AttributeError(s, name=None)
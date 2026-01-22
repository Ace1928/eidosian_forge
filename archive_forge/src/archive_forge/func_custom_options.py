import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
@classmethod
def custom_options(cls, val=None, backend=None):
    backend = cls.current_backend if backend is None else backend
    if val is None:
        return cls._custom_options[backend]
    else:
        cls._custom_options[backend] = val
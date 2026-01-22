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
def record_skipped_option(cls, error):
    """
        Record the OptionError associated with a skipped option if
        currently recording
        """
    if cls._errors_recorded is not None:
        cls._errors_recorded.append(error)
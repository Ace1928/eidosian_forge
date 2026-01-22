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
def validation_error_message(cls, spec, backends=None):
    """
        Returns an options validation error message if there are any
        invalid keywords. Otherwise returns None.
        """
    try:
        cls.validate_spec(spec, backends=backends)
    except OptionError as e:
        return e.format_options_error()
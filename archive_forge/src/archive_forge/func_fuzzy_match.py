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
def fuzzy_match(self, kw):
    """
        Given a string, fuzzy match against the Keyword values,
        returning a list of close matches.
        """
    return difflib.get_close_matches(kw, self.values)
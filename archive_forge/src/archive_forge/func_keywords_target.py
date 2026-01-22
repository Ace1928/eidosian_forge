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
def keywords_target(self, target):
    """
        Helper method to easily set the target on the allowed_keywords Keywords.
        """
    self.allowed_keywords.target = target
    return self
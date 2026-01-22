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
def capture_ids(cls, obj):
    """
        Given an list of ids, capture a list of ids that can be
        restored using the restore_ids.
        """
    return obj.traverse(lambda o: o.id)
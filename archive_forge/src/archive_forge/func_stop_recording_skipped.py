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
def stop_recording_skipped(cls):
    """
        Stop collecting OptionErrors recorded with the
        record_skipped_option method and return them
        """
    if cls._errors_recorded is None:
        raise Exception('Cannot stop recording before it is started')
    recorded = cls._errors_recorded[:]
    cls._errors_recorded = None
    return recorded
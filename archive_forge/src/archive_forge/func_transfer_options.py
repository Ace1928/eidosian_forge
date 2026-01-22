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
def transfer_options(cls, obj, new_obj, backend=None, names=None, level=3):
    """
        Transfers options for all backends from one object to another.
        Drops any options defined in the supplied drop list.
        """
    if obj is new_obj:
        return
    backend = cls.current_backend if backend is None else backend
    type_name = type(new_obj).__name__
    group = type_name if obj.group == type(obj).__name__ else obj.group
    spec = '.'.join([s for s in (type_name, group, obj.label)[:level] if s])
    options = []
    for group in Options._option_groups:
        opts = cls.lookup_options(backend, obj, group)
        if not opts:
            continue
        new_opts = cls.lookup_options(backend, new_obj, group, defaults=False)
        existing = new_opts.kwargs if new_opts else {}
        filtered = {k: v for k, v in opts.kwargs.items() if (names is None or k in names) and k not in existing}
        if filtered:
            options.append(Options(group, **filtered))
    if options:
        StoreOptions.set_options(new_obj, {spec: options}, backend)
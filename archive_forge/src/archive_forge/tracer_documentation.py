import warnings
from contextlib import contextmanager
from collections import defaultdict
from .util import subvals, toposort
from .wrap_util import wraps

    Wraps a function so that its gradient can be specified and its invocation
    can be recorded. For examples, see the docs.
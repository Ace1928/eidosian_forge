import bisect
import sys
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from itertools import islice
from operator import itemgetter
from time import time
from typing import Mapping, Optional  # noqa
from weakref import WeakSet, ref
from kombu.clocks import timetuple
from kombu.utils.objects import cached_property
from celery import states
from celery.utils.functional import LRUCache, memoize, pass1
from celery.utils.log import get_logger
def with_unique_field(attr):

    def _decorate_cls(cls):

        def __eq__(this, other):
            if isinstance(other, this.__class__):
                return getattr(this, attr) == getattr(other, attr)
            return NotImplemented
        cls.__eq__ = __eq__

        def __hash__(this):
            return hash(getattr(this, attr))
        cls.__hash__ = __hash__
        return cls
    return _decorate_cls
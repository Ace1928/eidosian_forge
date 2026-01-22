import numbers
from collections import namedtuple
from collections.abc import Mapping
from datetime import timedelta
from weakref import WeakValueDictionary
from kombu import Connection, Consumer, Exchange, Producer, Queue, pools
from kombu.common import Broadcast
from kombu.utils.functional import maybe_list
from kombu.utils.objects import cached_property
from celery import signals
from celery.utils.nodenames import anon_nodename
from celery.utils.saferepr import saferepr
from celery.utils.text import indent as textindent
from celery.utils.time import maybe_make_aware
from . import routes as _routes
def select_add(self, queue, **kwargs):
    """Add new task queue that'll be consumed from.

        The queue will be active even when a subset has been selected
        using the :option:`celery worker -Q` option.
        """
    q = self.add(queue, **kwargs)
    if self._consume_from is not None:
        self._consume_from[q.name] = q
    return q
import inspect
import os
import sys
import threading
import warnings
from collections import UserDict, defaultdict, deque
from datetime import datetime
from datetime import timezone as datetime_timezone
from operator import attrgetter
from click.exceptions import Exit
from dateutil.parser import isoparse
from kombu import pools
from kombu.clocks import LamportClock
from kombu.common import oid_from
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import starpromise
from celery import platforms, signals
from celery._state import (_announce_app_finalized, _deregister_app, _register_app, _set_current_app, _task_stack,
from celery.exceptions import AlwaysEagerIgnored, ImproperlyConfigured
from celery.loaders import get_loader_cls
from celery.local import PromiseProxy, maybe_evaluate
from celery.utils import abstract
from celery.utils.collections import AttributeDictMixin
from celery.utils.dispatch import Signal
from celery.utils.functional import first, head_from_fun, maybe_list
from celery.utils.imports import gen_task_name, instantiate, symbol_by_name
from celery.utils.log import get_logger
from celery.utils.objects import FallbackContext, mro_lookup
from celery.utils.time import maybe_make_aware, timezone, to_utc
from . import backends, builtins  # noqa
from .annotations import prepare as prepare_annotations
from .autoretry import add_autoretry_behaviour
from .defaults import DEFAULT_SECURITY_DIGEST, find_deprecated_settings
from .registry import TaskRegistry
from .utils import (AppPickler, Settings, _new_key_to_old, _old_key_to_new, _unpickle_app, _unpickle_app_v2, appstr,
def subclass_with_self(self, Class, name=None, attribute='app', reverse=None, keep_reduce=False, **kw):
    """Subclass an app-compatible class.

        App-compatible means that the class has a class attribute that
        provides the default app it should use, for example:
        ``class Foo: app = None``.

        Arguments:
            Class (type): The app-compatible class to subclass.
            name (str): Custom name for the target class.
            attribute (str): Name of the attribute holding the app,
                Default is 'app'.
            reverse (str): Reverse path to this object used for pickling
                purposes. For example, to get ``app.AsyncResult``,
                use ``"AsyncResult"``.
            keep_reduce (bool): If enabled a custom ``__reduce__``
                implementation won't be provided.
        """
    Class = symbol_by_name(Class)
    reverse = reverse if reverse else Class.__name__

    def __reduce__(self):
        return (_unpickle_appattr, (reverse, self.__reduce_args__()))
    attrs = dict({attribute: self}, __module__=Class.__module__, __doc__=Class.__doc__, **kw)
    if not keep_reduce:
        attrs['__reduce__'] = __reduce__
    return type(name or Class.__name__, (Class,), attrs)
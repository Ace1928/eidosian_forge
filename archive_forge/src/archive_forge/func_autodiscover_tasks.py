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
def autodiscover_tasks(self, packages=None, related_name='tasks', force=False):
    """Auto-discover task modules.

        Searches a list of packages for a "tasks.py" module (or use
        related_name argument).

        If the name is empty, this will be delegated to fix-ups (e.g., Django).

        For example if you have a directory layout like this:

        .. code-block:: text

            foo/__init__.py
               tasks.py
               models.py

            bar/__init__.py
                tasks.py
                models.py

            baz/__init__.py
                models.py

        Then calling ``app.autodiscover_tasks(['foo', 'bar', 'baz'])`` will
        result in the modules ``foo.tasks`` and ``bar.tasks`` being imported.

        Arguments:
            packages (List[str]): List of packages to search.
                This argument may also be a callable, in which case the
                value returned is used (for lazy evaluation).
            related_name (Optional[str]): The name of the module to find.  Defaults
                to "tasks": meaning "look for 'module.tasks' for every
                module in ``packages``.".  If ``None`` will only try to import
                the package, i.e. "look for 'module'".
            force (bool): By default this call is lazy so that the actual
                auto-discovery won't happen until an application imports
                the default modules.  Forcing will cause the auto-discovery
                to happen immediately.
        """
    if force:
        return self._autodiscover_tasks(packages, related_name)
    signals.import_modules.connect(starpromise(self._autodiscover_tasks, packages, related_name), weak=False, sender=self)
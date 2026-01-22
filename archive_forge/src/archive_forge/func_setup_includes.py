import os
import sys
from datetime import datetime, timezone
from billiard import cpu_count
from kombu.utils.compat import detect_environment
from celery import bootsteps
from celery import concurrency as _concurrency
from celery import signals
from celery.bootsteps import RUN, TERMINATE
from celery.exceptions import ImproperlyConfigured, TaskRevokedError, WorkerTerminate
from celery.platforms import EX_FAILURE, create_pidlock
from celery.utils.imports import reload_from_cwd
from celery.utils.log import mlevel
from celery.utils.log import worker_logger as logger
from celery.utils.nodenames import default_nodename, worker_direct
from celery.utils.text import str_to_list
from celery.utils.threads import default_socket_timeout
from . import state
def setup_includes(self, includes):
    prev = tuple(self.app.conf.include)
    if includes:
        prev += tuple(includes)
        [self.app.loader.import_task_module(m) for m in includes]
    self.include = includes
    task_modules = {task.__class__.__module__ for task in self.app.tasks.values()}
    self.app.conf.include = tuple(set(prev) | task_modules)
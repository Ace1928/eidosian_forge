import atexit
import os
import threading
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
import osprofiler.initializer
from glance.api import common
import glance.async_
from glance.common import config
from glance.common import store_utils
from glance import housekeeping
from glance.i18n import _, _LW
from glance import notifier
from glance import sqlite_migration
def drain_workers():
    pools_to_drain = ['tasks_pool']
    for pool_name in pools_to_drain:
        pool_model = common.get_thread_pool(pool_name)
        LOG.info('Waiting for remaining threads in pool %r', pool_name)
        pool_model.pool.shutdown()
    from glance.api.v2 import cached_images
    if cached_images.WORKER:
        cached_images.WORKER.terminate()
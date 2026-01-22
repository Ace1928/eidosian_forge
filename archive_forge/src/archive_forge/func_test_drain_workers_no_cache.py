from unittest import mock
from glance.api import common
from glance.api.v2 import cached_images
import glance.async_
from glance.common import exception
from glance.common import wsgi_app
from glance import sqlite_migration
from glance.tests import utils as test_utils
@mock.patch('glance.async_._THREADPOOL_MODEL', new=None)
def test_drain_workers_no_cache(self):
    glance.async_.set_threadpool_model('native')
    model = common.get_thread_pool('tasks_pool')
    with mock.patch.object(model.pool, 'shutdown'):
        wsgi_app.drain_workers()
        self.assertIsNone(cached_images.WORKER)
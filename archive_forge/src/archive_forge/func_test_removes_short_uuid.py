from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
def test_removes_short_uuid(self):
    id = uuid.uuid4().hex
    path = 'foo.{uuid}.bar'.format(uuid=id)
    stat = stats.StatsMiddleware.strip_short_uuid(path)
    self.assertEqual('foo.bar', stat)
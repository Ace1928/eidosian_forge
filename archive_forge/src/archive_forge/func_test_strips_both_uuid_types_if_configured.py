from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
def test_strips_both_uuid_types_if_configured(self):
    app = self.make_stats_middleware(remove_uuid=True, remove_short_uuid=True)
    random_short_uuid = uuid.uuid4().hex
    random_uuid = str(uuid.uuid4())
    path = '/foo/{uuid}/bar/{short_uuid}'.format(uuid=random_uuid, short_uuid=random_short_uuid)
    self.perform_request(app, path, 'GET')
    expected_stat = '{name}.{method}.foo.bar'.format(name=app.stat_name, method='GET')
    app.statsd.timer.assert_called_once_with(expected_stat)
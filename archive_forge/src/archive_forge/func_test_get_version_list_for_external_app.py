import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
def test_get_version_list_for_external_app(self):
    url = 'http://customhost:9292/app/api'
    req = webob.Request.blank('/', base_url=url)
    self.config(bind_host='127.0.0.1', bind_port=9292)
    res = versions.Controller().index(req)
    self.assertEqual(http.MULTIPLE_CHOICES, res.status_int)
    self.assertEqual('application/json', res.content_type)
    results = jsonutils.loads(res.body)['versions']
    expected = get_versions_list(url)
    self.assertEqual(expected, results)
    self.config(enabled_backends='slow:one,fast:two')
    res = versions.Controller().index(req)
    results = jsonutils.loads(res.body)['versions']
    expected = get_versions_list(url, enabled_backends=True)
    self.assertEqual(expected, results)
    self.config(image_cache_dir='/tmp/cache')
    res = versions.Controller().index(req)
    results = jsonutils.loads(res.body)['versions']
    expected = get_versions_list(url, enabled_backends=True, enabled_cache=True)
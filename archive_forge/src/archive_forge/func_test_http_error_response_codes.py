import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
def test_http_error_response_codes(self):
    sample_id, member_id, tag_val, task_id = ('abc', '123', '1', '2')
    'Makes sure v2 unallowed methods return 405'
    unallowed_methods = [('/schemas/image', ['POST', 'PUT', 'DELETE', 'PATCH', 'HEAD']), ('/schemas/images', ['POST', 'PUT', 'DELETE', 'PATCH', 'HEAD']), ('/schemas/member', ['POST', 'PUT', 'DELETE', 'PATCH', 'HEAD']), ('/schemas/members', ['POST', 'PUT', 'DELETE', 'PATCH', 'HEAD']), ('/schemas/task', ['POST', 'PUT', 'DELETE', 'PATCH', 'HEAD']), ('/schemas/tasks', ['POST', 'PUT', 'DELETE', 'PATCH', 'HEAD']), ('/images', ['PUT', 'DELETE', 'PATCH', 'HEAD']), ('/images/%s' % sample_id, ['POST', 'PUT', 'HEAD']), ('/images/%s/file' % sample_id, ['POST', 'DELETE', 'PATCH', 'HEAD']), ('/images/%s/tags/%s' % (sample_id, tag_val), ['GET', 'POST', 'PATCH', 'HEAD']), ('/images/%s/members' % sample_id, ['PUT', 'DELETE', 'PATCH', 'HEAD']), ('/images/%s/members/%s' % (sample_id, member_id), ['POST', 'PATCH', 'HEAD']), ('/tasks', ['PUT', 'DELETE', 'PATCH', 'HEAD']), ('/tasks/%s' % task_id, ['POST', 'PUT', 'PATCH', 'HEAD'])]
    api = test_utils.FakeAuthMiddleware(router_v2.API(routes.Mapper()))
    for uri, methods in unallowed_methods:
        for method in methods:
            req = webob.Request.blank(uri)
            req.method = method
            res = req.get_response(api)
            self.assertEqual(http.METHOD_NOT_ALLOWED, res.status_int)
    req = webob.Request.blank('/schemas/image')
    req.method = 'NonexistentMethod'
    res = req.get_response(api)
    self.assertEqual(http.METHOD_NOT_ALLOWED, res.status_int)
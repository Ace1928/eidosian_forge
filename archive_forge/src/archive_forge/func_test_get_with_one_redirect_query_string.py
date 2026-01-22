import http.client as http
import eventlet.patcher
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils
def test_get_with_one_redirect_query_string(self):
    """
        Test GET with one 302 FOUND redirect w/ a query string
        """
    response = self.client.do_request('GET', '/302', params={'with_qs': 'yes'})
    self.assertEqual(http.OK, response.status)
    self.assertEqual(b'success_with_qs', response.read())
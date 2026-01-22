from unittest import mock
from webob import response as webob_response
from osprofiler import _utils as utils
from osprofiler import profiler
from osprofiler.tests import test
from osprofiler import web
@mock.patch('osprofiler.web.profiler.init')
def test_wsgi_middleware_enable_via_python(self, mock_profiler_init):
    request = mock.MagicMock()
    request.get_response.return_value = 'yeah!'
    request.url = 'someurl'
    request.host_url = 'someurl'
    request.path = 'path'
    request.query_string = 'query'
    request.method = 'method'
    request.scheme = 'scheme'
    hmac_key = 'super_secret_key2'
    pack = utils.signed_pack({'base_id': '1', 'parent_id': '2'}, hmac_key)
    request.headers = {'a': '1', 'b': '2', 'X-Trace-Info': pack[0], 'X-Trace-HMAC': pack[1]}
    web.enable('super_secret_key1,super_secret_key2')
    middleware = web.WsgiMiddleware('app', enabled=True)
    self.assertEqual('yeah!', middleware(request))
    mock_profiler_init.assert_called_once_with(hmac_key=hmac_key, base_id='1', parent_id='2')
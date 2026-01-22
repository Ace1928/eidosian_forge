from oslo_config import fixture
from oslotest import base as test_base
import webob
import webob.dec
import webob.exc as exc
from oslo_middleware import cors
@webob.dec.wsgify
def test_application(req):
    if req.path_info == '/server_cors':
        response = webob.Response(status=200)
        response.headers['Access-Control-Allow-Origin'] = req.headers['Origin']
        response.headers['X-Server-Generated-Response'] = '1'
        return response
    if req.path_info == '/server_cors_vary':
        response = webob.Response(status=200)
        response.headers['Vary'] = 'Custom-Vary'
        return response
    if req.path_info == '/server_no_cors':
        response = webob.Response(status=200)
        return response
    if req.method == 'OPTIONS':
        raise exc.HTTPNotFound()
    return 'Hello World'
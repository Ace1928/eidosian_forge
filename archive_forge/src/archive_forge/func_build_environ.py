import io
import math
import sys
import typing
import warnings
import anyio
from anyio.abc import ObjectReceiveStream, ObjectSendStream
from starlette.types import Receive, Scope, Send
def build_environ(scope: Scope, body: bytes) -> typing.Dict[str, typing.Any]:
    """
    Builds a scope and request body into a WSGI environ object.
    """
    script_name = scope.get('root_path', '').encode('utf8').decode('latin1')
    path_info = scope['path'].encode('utf8').decode('latin1')
    if path_info.startswith(script_name):
        path_info = path_info[len(script_name):]
    environ = {'REQUEST_METHOD': scope['method'], 'SCRIPT_NAME': script_name, 'PATH_INFO': path_info, 'QUERY_STRING': scope['query_string'].decode('ascii'), 'SERVER_PROTOCOL': f'HTTP/{scope['http_version']}', 'wsgi.version': (1, 0), 'wsgi.url_scheme': scope.get('scheme', 'http'), 'wsgi.input': io.BytesIO(body), 'wsgi.errors': sys.stdout, 'wsgi.multithread': True, 'wsgi.multiprocess': True, 'wsgi.run_once': False}
    server = scope.get('server') or ('localhost', 80)
    environ['SERVER_NAME'] = server[0]
    environ['SERVER_PORT'] = server[1]
    if scope.get('client'):
        environ['REMOTE_ADDR'] = scope['client'][0]
    for name, value in scope.get('headers', []):
        name = name.decode('latin1')
        if name == 'content-length':
            corrected_name = 'CONTENT_LENGTH'
        elif name == 'content-type':
            corrected_name = 'CONTENT_TYPE'
        else:
            corrected_name = f'HTTP_{name}'.upper().replace('-', '_')
        value = value.decode('latin1')
        if corrected_name in environ:
            value = environ[corrected_name] + ',' + value
        environ[corrected_name] = value
    return environ
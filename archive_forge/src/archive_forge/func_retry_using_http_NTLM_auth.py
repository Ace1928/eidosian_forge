import warnings
import base64
import typing as t
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import UnsupportedAlgorithm
from requests.auth import AuthBase
from requests.packages.urllib3.response import HTTPResponse
import spnego
def retry_using_http_NTLM_auth(self, auth_header_field, auth_header, response, auth_type, args):
    server_certificate_hash = self._get_server_cert(response)
    cbt = None
    if server_certificate_hash:
        cbt = spnego.channel_bindings.GssChannelBindings(application_data=b'tls-server-end-point:' + server_certificate_hash)
    'Attempt to authenticate using HTTP NTLM challenge/response.'
    if auth_header in response.request.headers:
        return response
    content_length = int(response.request.headers.get('Content-Length', '0'), base=10)
    if hasattr(response.request.body, 'seek'):
        if content_length > 0:
            response.request.body.seek(-content_length, 1)
        else:
            response.request.body.seek(0, 0)
    response.content
    response.raw.release_conn()
    request = response.request.copy()
    client = spnego.client(self.username, self.password, protocol='ntlm', channel_bindings=cbt)
    negotiate_message = base64.b64encode(client.step()).decode()
    auth = '%s %s' % (auth_type, negotiate_message)
    request.headers[auth_header] = auth
    args_nostream = dict(args, stream=False)
    response2 = response.connection.send(request, **args_nostream)
    response2.content
    response2.raw.release_conn()
    request = response2.request.copy()
    if response2.headers.get('set-cookie'):
        request.headers['Cookie'] = response2.headers.get('set-cookie')
    auth_header_value = response2.headers[auth_header_field]
    auth_strip = auth_type + ' '
    ntlm_header_value = next((s for s in (val.lstrip() for val in auth_header_value.split(',')) if s.startswith(auth_strip))).strip()
    val = base64.b64decode(ntlm_header_value[len(auth_strip):].encode())
    authenticate_message = base64.b64encode(client.step(val)).decode()
    auth = '%s %s' % (auth_type, authenticate_message)
    request.headers[auth_header] = auth
    response3 = response2.connection.send(request, **args)
    response3.history.append(response)
    response3.history.append(response2)
    self.session_security = ShimSessionSecurity(client)
    return response3
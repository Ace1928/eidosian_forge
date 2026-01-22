from __future__ import annotations
import typing as t
from http import client
from urllib.parse import quote
from urllib.parse import urlsplit
from ..datastructures import EnvironHeaders
from ..http import is_hop_by_hop_header
from ..wsgi import get_input_stream
def proxy_to(self, opts: dict[str, t.Any], path: str, prefix: str) -> WSGIApplication:
    target = urlsplit(opts['target'])
    host = target.hostname.encode('idna').decode('ascii')

    def application(environ: WSGIEnvironment, start_response: StartResponse) -> t.Iterable[bytes]:
        headers = list(EnvironHeaders(environ).items())
        headers[:] = [(k, v) for k, v in headers if not is_hop_by_hop_header(k) and k.lower() not in ('content-length', 'host')]
        headers.append(('Connection', 'close'))
        if opts['host'] == '<auto>':
            headers.append(('Host', host))
        elif opts['host'] is None:
            headers.append(('Host', environ['HTTP_HOST']))
        else:
            headers.append(('Host', opts['host']))
        headers.extend(opts['headers'].items())
        remote_path = path
        if opts['remove_prefix']:
            remote_path = remote_path[len(prefix):].lstrip('/')
            remote_path = f'{target.path.rstrip('/')}/{remote_path}'
        content_length = environ.get('CONTENT_LENGTH')
        chunked = False
        if content_length not in ('', None):
            headers.append(('Content-Length', content_length))
        elif content_length is not None:
            headers.append(('Transfer-Encoding', 'chunked'))
            chunked = True
        try:
            if target.scheme == 'http':
                con = client.HTTPConnection(host, target.port or 80, timeout=self.timeout)
            elif target.scheme == 'https':
                con = client.HTTPSConnection(host, target.port or 443, timeout=self.timeout, context=opts['ssl_context'])
            else:
                raise RuntimeError(f"Target scheme must be 'http' or 'https', got {target.scheme!r}.")
            con.connect()
            remote_url = quote(remote_path, safe="!$&'()*+,/:;=@%")
            querystring = environ['QUERY_STRING']
            if querystring:
                remote_url = f'{remote_url}?{querystring}'
            con.putrequest(environ['REQUEST_METHOD'], remote_url, skip_host=True)
            for k, v in headers:
                if k.lower() == 'connection':
                    v = 'close'
                con.putheader(k, v)
            con.endheaders()
            stream = get_input_stream(environ)
            while True:
                data = stream.read(self.chunk_size)
                if not data:
                    break
                if chunked:
                    con.send(b'%x\r\n%s\r\n' % (len(data), data))
                else:
                    con.send(data)
            resp = con.getresponse()
        except OSError:
            from ..exceptions import BadGateway
            return BadGateway()(environ, start_response)
        start_response(f'{resp.status} {resp.reason}', [(k.title(), v) for k, v in resp.getheaders() if not is_hop_by_hop_header(k)])

        def read() -> t.Iterator[bytes]:
            while True:
                try:
                    data = resp.read(self.chunk_size)
                except OSError:
                    break
                if not data:
                    break
                yield data
        return read()
    return application
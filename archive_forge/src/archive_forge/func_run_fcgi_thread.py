from paste.deploy.converters import aslist, asbool
from paste.script.serve import ensure_port_cleanup
import warnings
def run_fcgi_thread(wsgi_app, global_conf, host=None, port=None, socket=None, umask=None, multiplexed=False):
    import flup.server.fcgi
    warn('fcgi_thread')
    if socket:
        assert host is None and port is None
        sock = socket
    elif host:
        assert host is not None and port is not None
        sock = (host, int(port))
        ensure_port_cleanup([sock])
    else:
        sock = None
    if umask is not None:
        umask = int(umask)
    s = flup.server.fcgi.WSGIServer(wsgi_app, bindAddress=sock, umask=umask, multiplexed=asbool(multiplexed))
    s.run()
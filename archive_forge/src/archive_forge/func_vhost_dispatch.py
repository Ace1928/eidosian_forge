import string
import sys
import types
import cherrypy
def vhost_dispatch(path_info):
    request = cherrypy.serving.request
    header = request.headers.get
    domain = header('Host', '')
    if use_x_forwarded_host:
        domain = header('X-Forwarded-Host', domain)
    prefix = domains.get(domain, '')
    if prefix:
        path_info = httputil.urljoin(prefix, path_info)
    result = next_dispatcher(path_info)
    section = request.config.get('tools.staticdir.section')
    if section:
        section = section[len(prefix):]
        request.config['tools.staticdir.section'] = section
    return result
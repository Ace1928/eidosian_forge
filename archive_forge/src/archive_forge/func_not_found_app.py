import html
import re
import os
from collections.abc import MutableMapping as DictMixin
from paste import httpexceptions
def not_found_app(self, environ, start_response):
    mapper = environ.get('paste.urlmap_object')
    if mapper:
        matches = [p for p, a in mapper.applications]
        extra = 'defined apps: %s' % ',\n  '.join(map(repr, matches))
    else:
        extra = ''
    extra += '\nSCRIPT_NAME: %r' % html.escape(environ.get('SCRIPT_NAME'))
    extra += '\nPATH_INFO: %r' % html.escape(environ.get('PATH_INFO'))
    extra += '\nHTTP_HOST: %r' % html.escape(environ.get('HTTP_HOST'))
    app = httpexceptions.HTTPNotFound(environ['PATH_INFO'], comment=html.escape(extra)).wsgi_application
    return app(environ, start_response)
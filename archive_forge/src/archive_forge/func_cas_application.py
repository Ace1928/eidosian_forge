from urllib.parse import urlencode
from urllib.request import urlopen
from paste.request import construct_url
from paste.httpexceptions import HTTPSeeOther, HTTPForbidden
def cas_application(environ, start_response):
    username = environ.get('REMOTE_USER', '')
    if username:
        return application(environ, start_response)
    qs = environ.get('QUERY_STRING', '').split('&')
    if qs and qs[-1].startswith('ticket='):
        ticket = qs.pop().split('=', 1)[1]
        environ['QUERY_STRING'] = '&'.join(qs)
        service = construct_url(environ)
        args = urlencode({'service': service, 'ticket': ticket})
        requrl = authority + 'validate?' + args
        result = urlopen(requrl).read().split('\n')
        if 'yes' == result[0]:
            environ['REMOTE_USER'] = result[1]
            environ['AUTH_TYPE'] = 'cas'
            return application(environ, start_response)
        exce = CASLoginFailure()
    else:
        service = construct_url(environ)
        args = urlencode({'service': service})
        location = authority + 'login?' + args
        exce = CASAuthenticate(location)
    return exce.wsgi_application(environ, start_response)
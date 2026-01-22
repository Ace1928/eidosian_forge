import cgi
import urlparse
import re
import paste.request
from paste import httpexceptions
from openid.store import filestore
from openid.consumer import consumer
from openid.oidutil import appendArgs
def do_process(self, request):
    """Handle the redirect from the OpenID server.
        """
    oidconsumer = self.oidconsumer
    token = request['query'].get('token', '')
    status, info = oidconsumer.completeAuth(token, request['query'])
    css_class = 'error'
    openid_url = None
    if status == consumer.FAILURE and info:
        openid_url = info
        fmt = 'Verification of %s failed.'
        message = fmt % (cgi.escape(openid_url),)
    elif status == consumer.SUCCESS:
        css_class = 'alert'
        if info:
            openid_url = info
            if self.url_to_username:
                username = self.url_to_username(request['environ'], openid_url)
            else:
                username = openid_url
            if 'paste.auth_tkt.set_user' in request['environ']:
                request['environ']['paste.auth_tkt.set_user'](username)
            if not self.login_redirect:
                fmt = 'If you had supplied a login redirect path, you would have been redirected there.  You have successfully verified %s as your identity.'
                message = fmt % (cgi.escape(openid_url),)
            else:
                request['environ']['paste.auth.open_id'] = openid_url
                request['environ']['PATH_INFO'] = self.login_redirect
                return self.app(request['environ'], request['start'])
        else:
            message = 'Verification cancelled'
    else:
        message = 'Verification failed.'
    return self.render(request, message, css_class, openid_url)
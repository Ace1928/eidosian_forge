import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
def logout_user_cookie(self, environ):
    cur_domain = environ.get('HTTP_HOST', environ.get('SERVER_NAME'))
    wild_domain = '.' + cur_domain
    expires = 'Sat, 01-Jan-2000 12:00:00 GMT'
    cookies = [('Set-Cookie', '%s=""; Expires="%s"; Path=/' % (self.cookie_name, expires)), ('Set-Cookie', '%s=""; Expires="%s"; Path=/; Domain=%s' % (self.cookie_name, expires, cur_domain)), ('Set-Cookie', '%s=""; Expires="%s"; Path=/; Domain=%s' % (self.cookie_name, expires, wild_domain))]
    return cookies
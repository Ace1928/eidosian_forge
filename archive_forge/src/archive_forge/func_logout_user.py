import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
def logout_user():
    set_cookies.extend(self.logout_user_cookie(environ))
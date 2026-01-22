import time as time_mod
from http.cookies import SimpleCookie
from urllib.parse import quote as url_quote
from urllib.parse import unquote as url_unquote
from paste import request
def cookie_setting_start_response(status, headers, exc_info=None):
    headers.extend(set_cookies)
    return start_response(status, headers, exc_info)
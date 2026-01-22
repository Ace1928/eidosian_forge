import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def get_ha1_dict_plain(user_password_dict):
    """Returns a get_ha1 function which obtains a plaintext password from a
    dictionary of the form: {username : password}.

    If you want a simple dictionary-based authentication scheme, with plaintext
    passwords, use get_ha1_dict_plain(my_userpass_dict) as the value for the
    get_ha1 argument to digest_auth().
    """

    def get_ha1(realm, username):
        password = user_password_dict.get(username)
        if password:
            return md5_hex('%s:%s:%s' % (username, realm, password))
        return None
    return get_ha1
import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def md5_hex(s):
    return md5(ntob(s, 'utf-8')).hexdigest()
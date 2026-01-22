import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def get_ha1(realm, username):
    result = None
    with open(filename, 'r') as f:
        for line in f:
            u, r, ha1 = line.rstrip().split(':')
            if u == username and r == realm:
                result = ha1
                break
    return result
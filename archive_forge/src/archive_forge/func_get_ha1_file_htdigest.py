import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def get_ha1_file_htdigest(filename):
    """Returns a get_ha1 function which obtains a HA1 password hash from a
    flat file with lines of the same format as that produced by the Apache
    htdigest utility. For example, for realm 'wonderland', username 'alice',
    and password '4x5istwelve', the htdigest line would be::

        alice:wonderland:3238cdfe91a8b2ed8e39646921a02d4c

    If you want to use an Apache htdigest file as the credentials store,
    then use get_ha1_file_htdigest(my_htdigest_file) as the value for the
    get_ha1 argument to digest_auth().  It is recommended that the filename
    argument be an absolute path, to avoid problems.
    """

    def get_ha1(realm, username):
        result = None
        with open(filename, 'r') as f:
            for line in f:
                u, r, ha1 = line.rstrip().split(':')
                if u == username and r == realm:
                    result = ha1
                    break
        return result
    return get_ha1
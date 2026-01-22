from hashlib import md5
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.lib import auth_basic
from cherrypy.test import helper
def checkpasshash_u(realm, user, password):
    p = userhashdict_u.get(user)
    return p and p == md5(ntob(password, 'utf-8')).hexdigest() or False
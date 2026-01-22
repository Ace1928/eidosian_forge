import binascii
import unicodedata
import base64
import cherrypy
from cherrypy._cpcompat import ntou, tonative
def checkpassword(realm, user, password):
    p = user_password_dict.get(user)
    return p and p == password or False
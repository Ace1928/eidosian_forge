import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
@cherrypy.expose
def cookies_and_headers(self):
    cherrypy.response.cookie['candy'] = 'bar'
    cherrypy.response.cookie['candy']['domain'] = 'cherrypy.dev'
    cherrypy.response.headers['Some-Header'] = 'My dÃ¶g has fleas'
    cherrypy.response.headers['Bytes-Header'] = b'Bytes given header'
    return 'Any content'
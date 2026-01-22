import cherrypy
from cherrypy import _json as json
from cherrypy._cpcompat import text_or_bytes, ntou
def json_handler(*args, **kwargs):
    value = cherrypy.serving.request._json_inner_handler(*args, **kwargs)
    return json.encode(value)
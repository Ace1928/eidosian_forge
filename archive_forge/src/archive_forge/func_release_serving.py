import os
import cherrypy
from cherrypy import _cpconfig, _cplogging, _cprequest, _cpwsgi, tools
from cherrypy.lib import httputil, reprconf
def release_serving(self):
    """Release the current serving (request and response)."""
    req = cherrypy.serving.request
    cherrypy.engine.publish('after_request')
    try:
        req.close()
    except Exception:
        cherrypy.log(traceback=True, severity=40)
    cherrypy.serving.clear()
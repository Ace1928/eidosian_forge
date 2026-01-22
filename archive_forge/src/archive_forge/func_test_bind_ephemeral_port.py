import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def test_bind_ephemeral_port(self):
    """
        A server configured to bind to port 0 will bind to an ephemeral
        port and indicate that port number on startup.
        """
    cherrypy.config.reset()
    bind_ephemeral_conf = {'server.socket_port': 0}
    cherrypy.config.update(bind_ephemeral_conf)
    cherrypy.engine.start()
    assert cherrypy.server.bound_addr != cherrypy.server.bind_addr
    _host, port = cherrypy.server.bound_addr
    assert port > 0
    cherrypy.engine.stop()
    assert cherrypy.server.bind_addr == cherrypy.server.bound_addr
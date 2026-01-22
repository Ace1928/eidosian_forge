import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def login_redir():
    if not getattr(cherrypy.request, 'login', None):
        raise cherrypy.InternalRedirect('/internalredirect/login')
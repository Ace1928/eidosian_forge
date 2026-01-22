import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def test_redirect_with_xss(self):
    """A redirect to a URL with HTML injected should result
        in page contents escaped."""
    self.getPage('/redirect/url_with_xss')
    self.assertStatus(303)
    assert b'<script>' not in self.body
    assert b'&lt;script&gt;' in self.body
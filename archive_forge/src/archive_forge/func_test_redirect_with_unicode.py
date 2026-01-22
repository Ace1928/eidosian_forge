import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def test_redirect_with_unicode(self):
    """
        A redirect to a URL with Unicode should return a Location
        header containing that Unicode URL.
        """
    return
    self.getPage('/redirect/url_with_unicode')
    self.assertStatus(303)
    loc = self.assertHeader('Location')
    assert ntou('тест', encoding='utf-8') in loc
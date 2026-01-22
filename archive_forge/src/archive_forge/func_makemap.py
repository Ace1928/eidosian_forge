import gzip
import io
import sys
import time
import types
import unittest
import operator
from http.client import IncompleteRead
import cherrypy
from cherrypy import tools
from cherrypy._cpcompat import ntou
from cherrypy.test import helper, _test_decorators
def makemap():
    m = self._merged_args().get('map', {})
    cherrypy.request.numerify_map = list(m.items())
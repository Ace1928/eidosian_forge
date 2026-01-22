import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
def db_namespace(self, k, v):
    if k == 'scheme':
        self.db = v
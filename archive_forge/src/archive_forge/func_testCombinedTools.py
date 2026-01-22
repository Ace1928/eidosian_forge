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
def testCombinedTools(self):
    expectedResult = (ntou('Hello,world') + europoundUnicode).encode('utf-8')
    zbuf = io.BytesIO()
    zfile = gzip.GzipFile(mode='wb', fileobj=zbuf, compresslevel=9)
    zfile.write(expectedResult)
    zfile.close()
    self.getPage('/euro', headers=[('Accept-Encoding', 'gzip'), ('Accept-Charset', 'ISO-8859-1,utf-8;q=0.7,*;q=0.7')])
    self.assertInBody(zbuf.getvalue()[:3])
    if not HAS_GZIP_COMPRESSION_HEADER_FIXED:
        return
    zbuf = io.BytesIO()
    zfile = gzip.GzipFile(mode='wb', fileobj=zbuf, compresslevel=6)
    zfile.write(expectedResult)
    zfile.close()
    self.getPage('/decorated_euro', headers=[('Accept-Encoding', 'gzip')])
    self.assertInBody(zbuf.getvalue()[:3])
    self.getPage('/decorated_euro/subpath', headers=[('Accept-Encoding', 'gzip')])
    self.assertInBody(bytes([(x + 3) % 256 for x in zbuf.getvalue()]))
import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
def testRanges(self):
    self.getPage('/ranges/get_ranges?bytes=3-6')
    self.assertBody('[(3, 7)]')
    self.getPage('/ranges/get_ranges?bytes=2-4,-1')
    self.assertBody('[(2, 5), (7, 8)]')
    self.getPage('/ranges/get_ranges?bytes=-100')
    self.assertBody('[(0, 8)]')
    if cherrypy.server.protocol_version == 'HTTP/1.1':
        self.getPage('/ranges/slice_file', [('Range', 'bytes=2-5')])
        self.assertStatus(206)
        self.assertHeader('Content-Type', 'text/html;charset=utf-8')
        self.assertHeader('Content-Range', 'bytes 2-5/14')
        self.assertBody('llo,')
        self.getPage('/ranges/slice_file', [('Range', 'bytes=4-6,2-5')])
        self.assertStatus(206)
        ct = self.assertHeader('Content-Type')
        expected_type = 'multipart/byteranges; boundary='
        assert ct.startswith(expected_type)
        boundary = ct[len(expected_type):]
        expected_body = '\r\n--%s\r\nContent-type: text/html\r\nContent-range: bytes 4-6/14\r\n\r\no, \r\n--%s\r\nContent-type: text/html\r\nContent-range: bytes 2-5/14\r\n\r\nllo,\r\n--%s--\r\n' % (boundary, boundary, boundary)
        self.assertBody(expected_body)
        self.assertHeader('Content-Length')
        self.getPage('/ranges/slice_file', [('Range', 'bytes=2300-2900')])
        self.assertStatus(416)
        self.assertHeader('Content-Range', 'bytes */14')
    elif cherrypy.server.protocol_version == 'HTTP/1.0':
        self.getPage('/ranges/slice_file', [('Range', 'bytes=2-5')])
        self.assertStatus(200)
        self.assertBody('Hello, world\r\n')
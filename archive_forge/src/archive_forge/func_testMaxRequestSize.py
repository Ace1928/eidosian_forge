import os
import cherrypy
from cherrypy.test import helper
def testMaxRequestSize(self):
    if getattr(cherrypy.server, 'using_apache', False):
        return self.skip('skipped due to known Apache differences... ')
    for size in (500, 5000, 50000):
        self.getPage('/', headers=[('From', 'x' * 500)])
        self.assertStatus(413)
    lines256 = 'x' * 248
    self.getPage('/', headers=[('Host', '%s:%s' % (self.HOST, self.PORT)), ('From', lines256)])
    cd = 'Content-Disposition: form-data; name="file"; filename="hello.txt"'
    body = '\r\n'.join(['--x', cd, 'Content-Type: text/plain', '', '%s', '--x--'])
    partlen = 200 - len(body)
    b = body % ('x' * partlen)
    h = [('Content-type', 'multipart/form-data; boundary=x'), ('Content-Length', '%s' % len(b))]
    self.getPage('/upload', h, 'POST', b)
    self.assertBody('Size: %d' % partlen)
    b = body % ('x' * 200)
    h = [('Content-type', 'multipart/form-data; boundary=x'), ('Content-Length', '%s' % len(b))]
    self.getPage('/upload', h, 'POST', b)
    self.assertStatus(413)
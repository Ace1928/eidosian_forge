import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.test import helper
def test_multipart_form_data(self):
    body = '\r\n'.join(['--X', 'Content-Disposition: form-data; name="foo"', '', 'bar', '--X', 'Content-Disposition: form-data; name="baz"', '', '111', '--X', 'Content-Disposition: form-data; name="baz"', '', '333', '--X--'])
    (self.getPage('/multipart_form_data', method='POST', headers=[('Content-Type', 'multipart/form-data;boundary=X'), ('Content-Length', str(len(body)))], body=body),)
    self.assertBody(repr([('baz', [ntou('111'), ntou('333')]), ('foo', ntou('bar'))]))
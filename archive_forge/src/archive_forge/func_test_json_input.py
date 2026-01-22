import cherrypy
from cherrypy.test import helper
from cherrypy._json import json
def test_json_input(self):
    if json is None:
        self.skip('json not found ')
        return
    body = '[13, "c"]'
    headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(body)))]
    self.getPage('/json_post', method='POST', headers=headers, body=body)
    self.assertBody('ok')
    body = '[13, "c"]'
    headers = [('Content-Type', 'text/plain'), ('Content-Length', str(len(body)))]
    self.getPage('/json_post', method='POST', headers=headers, body=body)
    self.assertStatus(415, 'Expected an application/json content type')
    body = '[13, -]'
    headers = [('Content-Type', 'application/json'), ('Content-Length', str(len(body)))]
    self.getPage('/json_post', method='POST', headers=headers, body=body)
    self.assertStatus(400, 'Invalid JSON document')